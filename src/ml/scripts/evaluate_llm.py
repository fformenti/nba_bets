"""CLI entry point for scoring a fine-tuned adapter.

    make evaluate-llm LLM_RUN=nba-bets-2026-07-27

Charts land in ``outputs/llm/eval/<run_name>/`` as standalone HTML, so they can
be scp'd off a headless box and opened locally.
"""

import os

# Must be set before anything imports torch, otherwise it has no effect.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

from src.config.paths import (  # noqa: E402
    DEFAULT_TRAIN_LLM_CONFIG_PATH,
    LLM_EVAL_OUTPUTS_DIR,
)
from src.ml.config.loader import load_llm_training_config  # noqa: E402
from src.ml.tracking.mlflow_tracker import MLflowTracker  # noqa: E402
from src.ml.training.llm_eval import run_evaluation  # noqa: E402
from src.utils.logging_config import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)
setup_logging(level="INFO")


def main(
    run_name: str,
    config_path: Optional[Path] = None,
    split: Optional[str] = None,
    size: Optional[int] = None,
    revision: Optional[str] = None,
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

    output_dir = LLM_EVAL_OUTPUTS_DIR / run_name

    def _evaluate():
        return run_evaluation(
            config=config,
            run_name=run_name,
            split=split,
            size=size,
            revision=revision,
            output_dir=output_dir,
        )

    mlflow_config = config.tracking.mlflow
    if not mlflow_config.enabled:
        result = _evaluate()
    else:
        with MLflowTracker(
            experiment_name=mlflow_config.experiment_name,
            run_name=f"eval-{run_name}",
            tracking_uri=mlflow_config.tracking_uri,
        ) as tracker:
            result = _evaluate()
            tracker.set_tags({"train_run_name": run_name, "stage": "evaluation"})
            tracker.log_params(
                {
                    "hub_model_id": result["hub_model_id"],
                    "eval_split": result["split"],
                    "eval_size": result["size"],
                    "revision": revision or "main",
                }
            )
            tracker.log_metrics(
                {"eval_accuracy": result["accuracy"], "eval_n_correct": result["n_correct"]}
            )
            tracker.log_artifacts(str(output_dir), artifact_path="eval_charts")

    logger.info(f"Charts written to {output_dir}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned LoRA adapter")
    parser.add_argument("--config", type=Path, help="Path to configuration YAML file")
    parser.add_argument(
        "--run-name",
        required=True,
        help="Training run name, which identifies the adapter's Hub repo.",
    )
    parser.add_argument("--split", help="Dataset split to score (default: config value)")
    parser.add_argument("--size", type=int, help="Number of datapoints to score")
    parser.add_argument("--revision", help="Hub revision/branch of the adapter")
    args = parser.parse_args()

    main(
        run_name=args.run_name,
        config_path=args.config,
        split=args.split,
        size=args.size,
        revision=args.revision,
    )
