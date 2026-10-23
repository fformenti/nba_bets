"""QLoRA supervised fine-tuning of a causal LM on the nba-bets dataset.

The library half of ``src/cli/train_llm.py``.  Everything here is
config-driven so a run can be relaunched on any GPU box and, crucially,
*resumed* from the last checkpoint pushed to the Hugging Face Hub after a
dropped connection.
"""

import logging
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

# hub_strategy="checkpoint" pushes the newest full checkpoint - adapter,
# optimizer, scheduler, RNG and trainer state - under this folder, overwriting
# it on every save. "every_save" pushes only the adapter to the repo root, which
# is nothing a run can resume from.
RESUME_DIR = "last-checkpoint"


def get_hf_token() -> str:
    """Return the HF token, raising a message that names the variable.

    Passed explicitly to every Hub call: the variable's project-specific name
    means huggingface_hub never finds it on its own.
    """
    return require_env(
        HF_TOKEN_ENV_VAR,
        "A token is required to read the gated base model, the dataset and adapters, "
        "and needs write scope to push adapters and checkpoints to the Hub.",
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
    """Download the Hub repo's ``last-checkpoint/`` folder, if it has one.

    Returns the local checkpoint directory to hand to
    ``SFTTrainer.train(resume_from_checkpoint=...)``, or ``None`` when the repo
    does not exist or holds no resumable checkpoint.
    """
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError

    api = HfApi(token=token)
    try:
        files = api.list_repo_files(hub_model_id)
    except RepositoryNotFoundError:
        logger.info(f"No existing Hub repo {hub_model_id} - starting a fresh run.")
        return None

    if not any(f.startswith(f"{RESUME_DIR}/") for f in files):
        logger.info(
            f"Hub repo {hub_model_id} exists but has no {RESUME_DIR}/ folder "
            "- starting a fresh run."
        )
        return None

    logger.info(f"Found {RESUME_DIR}/ in {hub_model_id}; downloading to resume.")
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=hub_model_id,
        token=token,
        local_dir=str(output_dir),
        allow_patterns=[f"{RESUME_DIR}/*"],
    )
    return output_dir / RESUME_DIR


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


def _drop_int8_cast_notice(record: logging.LogRecord) -> bool:
    """Drop bitsandbytes' per-matmul "inputs will be cast ... to float16" notice.

    LLM.int8's matmul only runs in fp16, so with bf16 compute every quantized
    layer casts its input on every call and logs it each time. The cast is
    expected and harmless (test_run2 did the same); see
    project_knowledge/llm_training/quantization_8bit.md.
    """
    return not str(record.msg).startswith("MatMul8bitLt: inputs will be cast")


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
    # Only this one notice: bitsandbytes' other warnings still show. Adding the
    # same filter again is a no-op.
    logging.getLogger("bitsandbytes.autograd._functions").addFilter(_drop_int8_cast_notice)
    return BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=compute_dtype,
    )


def build_tokenizer(config: LLMTrainingConfig):
    """Load the tokenizer with right-side padding on the EOS token."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model, trust_remote_code=True, token=get_hf_token()
    )
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
        token=get_hf_token(),
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
    """Upper bound on each record's token count as trl trains it.

    trl tokenizes ``prompt + completion + EOS`` as one string (BOS included).
    Counting the prompt (with BOS), the completion and EOS separately can only
    add tokens, where the joint string merges across the boundary, so the result
    is a safe ``max_length``: it never truncates a record.
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
        tf32=training.tf32 or None,
        seed=training.seed,
        logging_steps=training.logging_steps,
        # transformers defaults this from its own logger level, so anything that
        # raises verbosity (TRANSFORMERS_VERBOSITY=error, a notebook calling
        # set_verbosity_error) silently hides the progress bar. Pin it on.
        disable_tqdm=False,
        save_strategy="steps",
        save_steps=training.save_steps,
        save_total_limit=training.save_total_limit,
        load_best_model_at_end=training.load_best_model_at_end,
        metric_for_best_model=training.metric_for_best_model,
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
        # Also pushes the full checkpoint to last-checkpoint/, which is what
        # find_resumable_checkpoint downloads after a dropped connection.
        hub_strategy="checkpoint",
        hub_model_id=build_hub_model_id(config, run_name),
        hub_private_repo=config.hub.private,
    )


def load_train_dataset(config: LLMTrainingConfig):
    """Load the training split, honouring ``data.max_train_samples``.

    ``data.train_sample_strategy: random`` makes the cap a seeded sample across
    all seasons instead of the oldest N games.
    """
    from src.ml.llm.dataset import load_hf_split

    random = config.data.train_sample_strategy == "random"
    return load_hf_split(
        config,
        config.data.train_split,
        max_samples=config.data.max_train_samples,
        shuffle_seed=config.training.seed if random else None,
    )


def load_eval_dataset(config: LLMTrainingConfig):
    """Load the in-training evaluation split, honouring ``data.max_eval_samples``.

    The sample is drawn with ``evaluation.seed``, not ``training.seed``, so a
    seed repeat (E7) scores its eval curve on the same games as the original.
    """
    from src.ml.llm.dataset import load_hf_split

    return load_hf_split(
        config,
        config.data.eval_split,
        max_samples=config.data.max_eval_samples,
        shuffle_seed=config.evaluation.seed,
    )


def first_completion_token_hits(is_top1, label_mask):
    """Count rows whose first completion token is predicted correctly.

    trl's fused LM head never builds the full logits: its forward returns, per
    position, whether the next-token label was the top-1 prediction
    (``is_top1``) and whether that label is scored at all (``label_mask``),
    both already shifted by one. With completion-only loss the first scored
    position is the first completion token - the sign of the point differential.

    Returns ``(correct, total)`` as 0-d tensors. Rows whose completion was
    truncated away by ``max_length`` have no scored label and are excluded.
    """
    import torch

    has_completion = label_mask.any(dim=1)
    # argmax over a 0/1 mask returns the first 1; rows without one are dropped below.
    first = label_mask.int().argmax(dim=1)
    rows = torch.arange(label_mask.size(0), device=label_mask.device)
    correct = (is_top1[rows, first].bool() & has_completion).sum()
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

    trl computes the loss from its fused LM head, whose output carries the
    per-token ``is_top1``/``label_mask`` this needs, but ``compute_loss`` keeps
    that output to itself. Asking for it with ``return_outputs=True`` makes trl
    rebuild full logits with a second forward pass on every step, so a forward
    hook catches the fused output on its way out instead.
    """
    import torch
    from trl import SFTTrainer

    class FirstTokenAccuracySFTTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            captured = []
            handle = model.register_forward_hook(lambda _module, _args, out: captured.append(out))
            try:
                result = super().compute_loss(
                    model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
                )
            finally:
                handle.remove()

            fused = next((o for o in reversed(captured) if getattr(o, "is_top1", None) is not None), None)
            # Without completion-only loss the first scored label is a prompt token.
            if fused is not None and self.args.completion_only_loss:
                with torch.no_grad():
                    correct, total = first_completion_token_hits(fused.is_top1, fused.label_mask)
                correct = self.accelerator.gather_for_metrics(correct).sum()
                total = self.accelerator.gather_for_metrics(total).sum()
                if total > 0:
                    mode = "train" if self.model.training else "eval"
                    self._metrics[mode]["first_token_accuracy"].append((correct / total).item())
            return result

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

    callbacks = []
    if config.training.early_stopping_patience is not None:
        from transformers import EarlyStoppingCallback

        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=config.training.early_stopping_patience)
        )

    trainer = build_trainer_class()(
        model=base_model,
        train_dataset=train,
        eval_dataset=evaluation,
        peft_config=build_lora_config(config),
        args=build_sft_config(config, run_name, output_dir, max_length),
        callbacks=callbacks,
    )

    result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    memory = peak_gpu_memory()
    # Through the trainer's callbacks, so W&B and MLflow both record it.
    trainer.log(memory)
    logger.info(
        f"Peak GPU memory: {memory['peak_gpu_allocated_gb']:.1f} GB allocated, "
        f"{memory['peak_gpu_reserved_gb']:.1f} GB reserved"
    )
    train_metrics = {**result.metrics, **memory}

    trainer.model.push_to_hub(hub_model_id, private=config.hub.private)
    logger.info(f"Pushed final adapter to the hub: {hub_model_id}")

    return {
        "run_name": run_name,
        "hub_model_id": hub_model_id,
        "resumed_from": str(checkpoint) if checkpoint else None,
        "train_metrics": train_metrics,
    }


def peak_gpu_memory() -> dict[str, float]:
    """Peak GPU memory since the process started, in GB.

    ``allocated`` is what tensors held; ``reserved`` adds the caching
    allocator's slack and is the one that hits the card's limit first.
    """
    import torch

    return {
        "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
    }
