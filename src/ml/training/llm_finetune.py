"""QLoRA supervised fine-tuning of a causal LM on the nba-bets dataset.

The library half of ``src/ml/scripts/train_llm.py``.  Everything here is
config-driven so a run can be relaunched on any GPU box and, crucially,
*resumed* from the last checkpoint pushed to the Hugging Face Hub after a
dropped connection.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.ml.config.schema import LLMTrainingConfig
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

HF_TOKEN_ENV_VAR = "HF_llm_training_token"
WANDB_TOKEN_ENV_VAR = "WANDB_API_KEY"

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)/")


def get_hf_token() -> str:
    """Return the HF write token, raising a message that names the variable."""
    token = os.getenv(HF_TOKEN_ENV_VAR)
    if not token:
        raise RuntimeError(
            f"{HF_TOKEN_ENV_VAR} is not set. Add it to .env (a token with write "
            "scope is required to push adapters and checkpoints to the Hub)."
        )
    return token


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


def find_resumable_checkpoint(
    hub_model_id: str, output_dir: Path, token: str
) -> Optional[Path]:
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
        f"GPU: {device_name} | free {free_bytes / 1e9:.1f} GB "
        f"of {total_bytes / 1e9:.1f} GB"
    )

    token = get_hf_token()

    if config.tracking.wandb.enabled and not os.getenv(WANDB_TOKEN_ENV_VAR):
        raise RuntimeError(
            f"{WANDB_TOKEN_ENV_VAR} is not set but tracking.wandb.enabled is true. "
            "Add the key to .env or set tracking.wandb.enabled: false."
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


def build_sft_config(config: LLMTrainingConfig, run_name: str, output_dir: Path):
    """Map the YAML config onto a trl ``SFTConfig``."""
    from trl import SFTConfig

    training = config.training
    return SFTConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        num_train_epochs=training.epochs,
        per_device_train_batch_size=training.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        eval_strategy="no",
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        gradient_checkpointing=training.gradient_checkpointing,
        optim=training.optimizer,
        learning_rate=training.learning_rate,
        lr_scheduler_type=training.lr_scheduler_type,
        warmup_ratio=training.warmup_ratio,
        weight_decay=training.weight_decay,
        max_grad_norm=training.max_grad_norm,
        max_steps=-1,
        bf16=training.bf16,
        fp16=training.fp16,
        seed=training.seed,
        logging_steps=training.logging_steps,
        save_strategy="steps",
        save_steps=training.save_steps,
        save_total_limit=training.save_total_limit,
        report_to=_report_to(config),
        dataset_text_field=config.data.text_field,
        # The notebook defined MAX_SEQUENCE_LENGTH but never passed it.
        max_length=config.data.max_sequence_length,
        push_to_hub=config.hub.push_checkpoints,
        hub_strategy="every_save",
        hub_model_id=build_hub_model_id(config, run_name),
        hub_private_repo=config.hub.private,
    )


def load_train_dataset(config: LLMTrainingConfig):
    """Load the training split, honouring ``data.max_train_samples``."""
    from datasets import load_dataset

    dataset = load_dataset(config.data.dataset_name, token=get_hf_token())
    train = dataset[config.data.train_split]
    if config.data.max_train_samples:
        n = min(config.data.max_train_samples, len(train))
        train = train.select(range(n))
        logger.info(f"Capped training set to {n} rows (max_train_samples).")
    logger.info(f"Training rows: {len(train)}")
    return train


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
    from trl import SFTTrainer

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

    train = load_train_dataset(config)
    tokenizer = build_tokenizer(config)
    base_model = build_base_model(config, tokenizer)

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train,
        peft_config=build_lora_config(config),
        args=build_sft_config(config, run_name, output_dir),
    )

    result = trainer.train(
        resume_from_checkpoint=str(checkpoint) if checkpoint else None
    )

    trainer.model.push_to_hub(hub_model_id, private=config.hub.private)
    logger.info(f"Pushed final adapter to the hub: {hub_model_id}")

    return {
        "run_name": run_name,
        "hub_model_id": hub_model_id,
        "resumed_from": str(checkpoint) if checkpoint else None,
        "train_metrics": dict(result.metrics),
    }
