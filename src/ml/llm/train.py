"""Orchestration for a QLoRA fine-tuning run: preflight, tracking, resume.

Launch it so it survives an SSH disconnect::

    nohup make train-llm LLM_RUN=nba-bets-2026-07-27 > train.log 2>&1 &

If the run dies, re-run the *identical* command: it finds the newest
checkpoint in the Hub repo and picks up from there.
"""

import os

from pathlib import Path
from typing import Optional

from src.config.paths import DEFAULT_TRAIN_LLM_CONFIG_PATH
from src.ml.config.loader import load_llm_training_config
from src.ml.tracking.mlflow_tracker import MLflowTracker
from src.ml.llm.finetune import (
    preflight,
    resolve_run_name,
    run_training,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _configure_wandb_env(config) -> None:
    """Point the HF wandb integration at the right project."""
    wandb_config = config.tracking.wandb
    if not wandb_config.enabled:
        return
    os.environ["WANDB_PROJECT"] = wandb_config.project
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    os.environ["WANDB_WATCH"] = wandb_config.watch


def run_llm_training(
    config_path: Optional[Path] = None,
    run_name: Optional[str] = None,
    resume: str = "auto",
    max_train_samples: Optional[int] = None,
):
    resolved = Path(config_path) if config_path else DEFAULT_TRAIN_LLM_CONFIG_PATH
    if not resolved.exists():
        raise FileNotFoundError(
            f"LLM training config not found: {resolved}. "
            "Pass config_path= or --config with a valid YAML "
            "(e.g. configs/train_llm/llama31_8b_qlora.yaml)."
        )
    config = load_llm_training_config(resolved)

    if max_train_samples is not None:
        config.data.max_train_samples = max_train_samples

    resolved_run_name = resolve_run_name(config, run_name)
    preflight(config)
    _configure_wandb_env(config)

    mlflow_config = config.tracking.mlflow
    if not mlflow_config.enabled:
        return _train_and_report(config, resolved_run_name, resume, tracker=None)

    # Open the MLflow run *before* the trainer is built so the HF MLflowCallback
    # attaches to it rather than starting a run of its own.
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_config.tracking_uri
    with MLflowTracker(
        experiment_name=mlflow_config.experiment_name,
        run_name=resolved_run_name,
        tracking_uri=mlflow_config.tracking_uri,
    ) as tracker:
        tracker.log_config(config.model_dump())
        return _train_and_report(config, resolved_run_name, resume, tracker)


def _train_and_report(config, run_name: str, resume: str, tracker):
    try:
        result = run_training(config=config, run_name=run_name, resume=resume)
    finally:
        if config.tracking.wandb.enabled:
            import wandb

            if wandb.run is not None:
                wandb.finish()

    if tracker:
        tracker.log_params(
            {
                "hub_model_id": result["hub_model_id"],
                "resumed_from": result["resumed_from"] or "scratch",
            }
        )
        numeric_metrics = {
            key: value
            for key, value in result["train_metrics"].items()
            if isinstance(value, (int, float))
        }
        tracker.log_metrics(numeric_metrics)

    logger.info("Training complete!")
    logger.info(f"  run name:     {result['run_name']}")
    logger.info(f"  hub model id: {result['hub_model_id']}")
    logger.info(
        f"  evaluate it:  make evaluate-llm LLM_RUN={result['run_name']}"
    )
    return result
