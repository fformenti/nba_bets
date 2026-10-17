"""QLoRA supervised fine-tuning of a causal LM on the nba-bets dataset.

The library half of ``src/cli/train_llm.py``.  Everything here is
config-driven so a run can be relaunched on any GPU box and, crucially,
*resumed* from the last checkpoint pushed to the Hugging Face Hub after a
dropped connection.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.config.secrets import (
    GLOBAL_SECRETS_FILE,
    HF_TOKEN_ENV_VAR,
    WANDB_TOKEN_ENV_VAR,
    require_env,
)
from src.ml.config.schema import LLMTrainingConfig
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)/")


def get_hf_token() -> str:
    """Return the HF write token, raising a message that names the variable."""
    return require_env(
        HF_TOKEN_ENV_VAR,
        "A token with write scope is required to push adapters and checkpoints to the Hub.",
    )


def resolve_run_name(config: LLMTrainingConfig, cli_run_name: Optional[str]) -> str:
    """Resolve the run name: CLI > config > freshly generated timestamp.

    The run name determines the Hub repo, so resuming requires the same value.
    """
    if cli_run_name:
        return cli_run_name
    if config.run_name:
        return config.run_name
    return f"{config.hub.project_name}-{datetime.now():%Y-%m-%d_%H.%M.%S}"


def build_hub_model_id(config: LLMTrainingConfig, run_name: str) -> str:
    """Fully qualified Hub repo id for this run's adapter."""
    return f"{config.hub.user}/{run_name}"


def find_resumable_checkpoint(hub_model_id: str, output_dir: Path, token: str) -> Optional[Path]:
    """Download the newest ``checkpoint-N`` from the Hub repo, if any.

    Returns the local checkpoint directory to hand to
    ``SFTTrainer.train(resume_from_checkpoint=...)``, or ``None`` when the repo
    does not exist or holds no checkpoints.
    """
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError

    api = HfApi(token=token)
    try:
        files = api.list_repo_files(hub_model_id)
    except RepositoryNotFoundError:
        logger.info(f"No existing Hub repo {hub_model_id} - starting a fresh run.")
        return None

    steps = {int(match.group(1)) for f in files if (match := _CHECKPOINT_RE.match(f))}
    if not steps:
        logger.info(
            f"Hub repo {hub_model_id} exists but has no checkpoint-* directories "
            "- starting a fresh run."
        )
        return None

    latest = max(steps)
    logger.info(f"Found checkpoint-{latest} in {hub_model_id}; downloading to resume.")
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=hub_model_id,
        token=token,
        local_dir=str(output_dir),
        allow_patterns=[f"checkpoint-{latest}/*"],
    )
    return output_dir / f"checkpoint-{latest}"


def preflight(config: LLMTrainingConfig) -> None:
    """Fail fast, before the multi-GB model download, on a misconfigured box.

    Checks GPU availability, required secrets and dataset reachability - all of
    which otherwise surface many minutes into a run.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device visible. This job is meant to run on a GPU box; "
            "check the driver install and that torch was built with CUDA."
        )

    device_name = torch.cuda.get_device_name(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    logger.info(
        f"GPU: {device_name} | free {free_bytes / 1e9:.1f} GB of {total_bytes / 1e9:.1f} GB"
    )

    token = get_hf_token()

    if config.tracking.wandb.enabled:
        require_env(
            WANDB_TOKEN_ENV_VAR,
            "Alternatively set tracking.wandb.enabled: false to train without it.",
            secrets_file=GLOBAL_SECRETS_FILE,
        )

    from huggingface_hub import HfApi

    HfApi(token=token).dataset_info(config.data.dataset_name)
    logger.info(f"Dataset {config.data.dataset_name} is reachable.")


def build_quant_config(config: LLMTrainingConfig):
    """Build the bitsandbytes config, or ``None`` for full-precision loading."""
    import torch
    from transformers import BitsAndBytesConfig

    quant = config.quantization
    if quant.mode == "none":
        return None

    compute_dtype = getattr(torch, quant.compute_dtype)
    if quant.mode == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=quant.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quant.bnb_4bit_quant_type,
        )
    return BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=compute_dtype,
    )


def build_tokenizer(config: LLMTrainingConfig):
    """Load the tokenizer with right-side padding on the EOS token."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_base_model(config: LLMTrainingConfig, tokenizer):
    """Load the (optionally quantized) base model onto the available devices."""
    from transformers import AutoModelForCausalLM

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=build_quant_config(config),
        device_map="auto",
    )
    base_model.generation_config.pad_token_id = tokenizer.pad_token_id
    logger.info(
        f"Memory footprint: {base_model.get_memory_footprint() / 1e6:.1f} MB "
        f"(quantization={config.quantization.mode})"
    )
    return base_model


def build_lora_config(config: LLMTrainingConfig):
    from peft import LoraConfig

    return LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        task_type="CAUSAL_LM",
    )


def _report_to(config: LLMTrainingConfig) -> list[str]:
    reporters = []
    if config.tracking.wandb.enabled:
        reporters.append("wandb")
    if config.tracking.mlflow.enabled:
        reporters.append("mlflow")
    return reporters


def sequence_lengths(prompts, completions, tokenizer) -> list[int]:
    """Token count of each prompt+completion, as trl tokenizes it for training.

    trl 0.22 tokenizes the two fields separately: the prompt with special tokens
    (BOS), the completion without, then appends EOS. Tokenizing the
    concatenated string can merge tokens across the boundary and undercount.
    """
    prompt_ids = tokenizer(list(prompts))["input_ids"]
    completion_ids = tokenizer(list(completions), add_special_tokens=False)["input_ids"]
    return [len(p) + len(c) + 1 for p, c in zip(prompt_ids, completion_ids)]


def check_context_length(
    lengths_by_split: dict[str, list[int]],
    game_ids_by_split: dict[str, list],
    context_length: int,
) -> int:
    """Return the longest record across all splits; raise if it exceeds the context window."""
    split, lengths = max(lengths_by_split.items(), key=lambda kv: max(kv[1]))
    longest = max(lengths)
    if longest > context_length:
        game_id = game_ids_by_split[split][lengths.index(longest)]
        raise ValueError(
            f"Record game_id={game_id} in split '{split}' is {longest} tokens, "
            f"over the model's context length of {context_length}."
        )
    return longest


def resolve_max_length(config: LLMTrainingConfig, tokenizer) -> int:
    """Measure prompt+completion lengths over every split and return the longest.

    Covers the whole dataset, ignoring the sample caps, so the cap never
    truncates a record any split contains.
    """
    import numpy as np
    from datasets import load_dataset

    dataset = load_dataset(config.data.dataset_name, token=get_hf_token())
    lengths_by_split, game_ids_by_split = {}, {}
    for name, split in dataset.items():
        lengths = sequence_lengths(split["prompt"], split["completion"], tokenizer)
        lengths_by_split[name] = lengths
        game_ids_by_split[name] = split["game_id"]
        logger.info(
            f"Token lengths ({name}): max={max(lengths)} "
            f"p99={int(np.percentile(lengths, 99))} rows={len(lengths)}"
        )

    longest = check_context_length(
        lengths_by_split, game_ids_by_split, config.data.context_length
    )
    logger.info(f"max_length set to {longest} (context length {config.data.context_length}).")
    return longest


def build_sft_config(
    config: LLMTrainingConfig, run_name: str, output_dir: Path, max_length: int
):
    """Map the YAML config onto a trl ``SFTConfig``."""
    from trl import SFTConfig

    training = config.training
    return SFTConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        num_train_epochs=training.epochs,
        per_device_train_batch_size=training.per_device_train_batch_size,
        per_device_eval_batch_size=training.per_device_eval_batch_size,
        eval_strategy=training.eval_strategy,
        eval_steps=training.eval_steps,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        gradient_checkpointing=training.gradient_checkpointing,
        optim=training.optimizer,
        learning_rate=training.learning_rate,
        lr_scheduler_type=training.lr_scheduler_type,
        # transformers>=5.x merged warmup_ratio into warmup_steps: a float <1 is
        # treated as a ratio of total steps, >=1 as an absolute step count.
        warmup_steps=training.warmup_ratio,
        weight_decay=training.weight_decay,
        max_grad_norm=training.max_grad_norm,
        max_steps=-1,
        bf16=training.bf16,
        fp16=training.fp16,
        seed=training.seed,
        logging_steps=training.logging_steps,
        # transformers defaults this from its own logger level, so anything that
        # raises verbosity (TRANSFORMERS_VERBOSITY=error, a notebook calling
        # set_verbosity_error) silently hides the progress bar. Pin it on.
        disable_tqdm=False,
        save_strategy="steps",
        save_steps=training.save_steps,
        save_total_limit=training.save_total_limit,
        report_to=_report_to(config),
        # No dataset_text_field on purpose. trl dispatches on column names, and
        # checks for a text field *before* prompt/completion: naming one here
        # that exists in the dataset routes to language modeling, which ignores
        # the completion column entirely and rejects completion_only_loss. The
        # dataset ships prompt+completion, so leaving this at its default (which
        # matches no column) is what selects the prompt-completion path.
        completion_only_loss=training.train_on_completion_only,
        # Measured from the dataset (resolve_max_length), so nothing is truncated.
        max_length=max_length,
        push_to_hub=config.hub.push_checkpoints,
        hub_strategy="every_save",
        hub_model_id=build_hub_model_id(config, run_name),
        hub_private_repo=config.hub.private,
    )


def load_train_dataset(config: LLMTrainingConfig):
    """Load the training split, honouring ``data.max_train_samples``."""
    from src.ml.llm.dataset import load_hf_split

    return load_hf_split(
        config, config.data.train_split, max_samples=config.data.max_train_samples
    )


def load_eval_dataset(config: LLMTrainingConfig):
    """Load the in-training evaluation split, honouring ``data.max_eval_samples``."""
    from src.ml.llm.dataset import load_hf_split

    return load_hf_split(
        config,
        config.data.eval_split,
        max_samples=config.data.max_eval_samples,
        shuffle_seed=config.training.seed,
    )


def first_completion_token_hits(logits, labels):
    """Count rows whose first completion token is predicted correctly.

    ``labels`` is the collator's output: prompt and padding positions are
    ``-100``, so with completion-only loss the first unmasked label is the
    first completion token - the sign of the point differential. Logits at
    position ``i`` predict the token at ``i + 1``, hence the one-step shift.

    Returns ``(correct, total)`` as 0-d tensors. Rows whose completion was
    truncated away by ``max_length`` have no unmasked label and are excluded.
    """
    import torch

    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    has_completion = mask.any(dim=1)
    # argmax over a 0/1 mask returns the first 1; rows without one are dropped below.
    first = mask.int().argmax(dim=1)
    rows = torch.arange(labels.size(0), device=labels.device)

    target = shift_labels[rows, first]
    predicted = logits[rows, first].argmax(dim=-1)
    correct = ((predicted == target) & has_completion).sum()
    return correct, has_completion.sum()


def build_trainer_class():
    """``SFTTrainer`` that also logs ``first_token_accuracy``.

    trl's own ``mean_token_accuracy`` averages over every completion token -
    sign, magnitude and the appended EOS - so it cannot tell a model that gets
    the winner right from one that only learned to stop. This adds the sign
    alone, in train and eval, through the same ``_metrics`` buffer trl uses,
    so it is averaged and logged (``eval_`` prefix included) alongside it.

    Built inside a function because trl is a GPU-only extra and every module
    under ``src/`` must import without it.
    """
    import torch
    from trl import SFTTrainer

    class FirstTokenAccuracySFTTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
            # Without completion-only loss the first unmasked label is a prompt
            # token; liger returns no logits.
            if (
                "labels" in inputs
                and self.args.completion_only_loss
                and not self.args.use_liger_kernel
            ):
                with torch.no_grad():
                    logits = outputs.logits[:, self.num_virtual_tokens :, :]
                    correct, total = first_completion_token_hits(logits, inputs["labels"])
                    correct = self.accelerator.gather_for_metrics(correct).sum()
                    total = self.accelerator.gather_for_metrics(total).sum()
                if total > 0:
                    mode = "train" if self.model.training else "eval"
                    self._metrics[mode]["first_token_accuracy"].append((correct / total).item())
            return (loss, outputs) if return_outputs else loss

    return FirstTokenAccuracySFTTrainer


def run_training(
    config: LLMTrainingConfig,
    run_name: str,
    resume: str = "auto",
) -> dict[str, Any]:
    """Fine-tune the base model and push the adapter to the Hub.

    Parameters
    ----------
    config : LLMTrainingConfig
        Validated configuration.
    run_name : str
        Resolved run name; also determines the Hub repo id.
    resume : {'auto', 'never'}
        ``auto`` resumes from the newest checkpoint in the Hub repo when one
        exists. ``never`` always starts from scratch.

    Returns
    -------
    dict
        ``run_name``, ``hub_model_id``, ``resumed_from`` and ``train_metrics``.
    """
    from huggingface_hub import login

    token = get_hf_token()
    login(token, add_to_git_credential=True)

    hub_model_id = build_hub_model_id(config, run_name)
    output_dir = Path(config.paths.output_dir) / run_name

    checkpoint = None
    if resume == "auto":
        checkpoint = find_resumable_checkpoint(hub_model_id, output_dir, token)
    else:
        logger.info("resume=never - starting from scratch.")

    logger.info(f"Run name:     {run_name}")
    logger.info(f"Hub model id: {hub_model_id}")
    logger.info(f"Output dir:   {output_dir}")
    logger.info(f"Resuming from: {checkpoint or 'scratch'}")

    tokenizer = build_tokenizer(config)
    # Before the base model loads, so an over-length record fails cheaply.
    max_length = resolve_max_length(config, tokenizer)
    train = load_train_dataset(config)
    evaluation = load_eval_dataset(config) if config.training.eval_strategy != "no" else None
    base_model = build_base_model(config, tokenizer)

    trainer = build_trainer_class()(
        model=base_model,
        train_dataset=train,
        eval_dataset=evaluation,
        peft_config=build_lora_config(config),
        args=build_sft_config(config, run_name, output_dir, max_length),
    )

    result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)

    trainer.model.push_to_hub(hub_model_id, private=config.hub.private)
    logger.info(f"Pushed final adapter to the hub: {hub_model_id}")

    return {
        "run_name": run_name,
        "hub_model_id": hub_model_id,
        "resumed_from": str(checkpoint) if checkpoint else None,
        "train_metrics": dict(result.metrics),
    }
