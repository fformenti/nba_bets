"""CLI entry point for QLoRA fine-tuning on a GPU box.

Launch it so it survives an SSH disconnect::

    nohup make train-llm LLM_RUN=nba-bets-2026-07-27 > train.log 2>&1 &

If the run dies, re-run the *identical* command: it finds the newest
checkpoint in the Hub repo and picks up from there.
"""

import os

# Must be set before anything imports torch, otherwise it has no effect.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

from src.config.paths import DEFAULT_TRAIN_LLM_CONFIG_PATH  # noqa: E402
from src.ml.config.loader import load_llm_training_config  # noqa: E402
from src.ml.tracking.mlflow_tracker import MLflowTracker  # noqa: E402
from src.ml.training.llm_finetune import (  # noqa: E402
    preflight,
    resolve_run_name,
    run_training,
)
from src.utils.logging_config import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)
setup_logging(level="INFO")


def _configure_wandb_env(config) -> None:
    """Point the HF wandb integration at the right project."""
    wandb_config = config.tracking.wandb
    if not wandb_config.enabled:
        return
    os.environ["WANDB_PROJECT"] = wandb_config.project
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    os.environ["WANDB_WATCH"] = wandb_config.watch


def main(
    config_path: Optional[Path] = None,
    run_name: Optional[str] = None,
    resume: str = "auto",
    max_train_samples: Optional[int] = None,
):
    setup_logging(level="INFO")
    load_dotenv()

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QLoRA fine-tune a causal LM")
    parser.add_argument("--config", type=Path, help="Path to configuration YAML file")
    parser.add_argument(
        "--run-name",
        help="Run name, which also names the Hub repo. Pass the name of an "
        "interrupted run to resume it.",
    )
    parser.add_argument(
        "--resume",
        choices=["auto", "never"],
        default="auto",
        help="auto (default) resumes from the newest Hub checkpoint if one exists.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        help="Cap training rows; overrides data.max_train_samples for smoke tests.",
    )
    args = parser.parse_args()

    main(
        config_path=args.config,
        run_name=args.run_name,
        resume=args.resume,
        max_train_samples=args.max_train_samples,
    )
