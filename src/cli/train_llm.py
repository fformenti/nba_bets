"""QLoRA fine-tune a causal LM on a GPU box.

Launch it so it survives an SSH disconnect::

    nohup make train-llm LLM_RUN=nba-bets-2026-07-27 > train.log 2>&1 &

If the run dies, re-run the *identical* command: it finds the newest checkpoint
in the Hub repo and picks up from there.
"""

import os

# Must be set before anything imports torch, otherwise it has no effect — hence
# the deferred imports below.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

from src.ml.llm.train import run_llm_training  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402


def main() -> None:
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

    setup_logging(level="INFO")
    run_llm_training(
        config_path=args.config,
        run_name=args.run_name,
        resume=args.resume,
        max_train_samples=args.max_train_samples,
    )


if __name__ == "__main__":
    main()
