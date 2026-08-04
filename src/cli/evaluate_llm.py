"""Score a fine-tuned LoRA adapter against the holdout split.

    make evaluate-llm LLM_RUN=nba-bets-2026-07-27

Charts land in ``outputs/llm/eval/<run_name>/`` as standalone HTML, so they can
be scp'd off a headless box and opened locally.
"""

import os

# Must be set before anything imports torch, otherwise it has no effect.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

from src.ml.llm.evaluate import run_tracked_evaluation  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402


def main() -> None:
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

    setup_logging(level="INFO")
    run_tracked_evaluation(
        run_name=args.run_name,
        config_path=args.config,
        split=args.split,
        size=args.size,
        revision=args.revision,
    )


if __name__ == "__main__":
    main()
