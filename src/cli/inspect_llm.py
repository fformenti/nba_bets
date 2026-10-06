"""Dump a fine-tuned adapter's raw answers and first-token sign probabilities.

    make inspect-llm LLM_RUN=nba-bets-2026-07-27

Writes ``outputs/llm/eval/<run_name>/inspection_<split>.csv`` - one row per
game with the raw decoded text, the parsed number, P(+), P(-) and the top
first tokens - and prints a summary. ``--summarize-only`` re-summarizes a
saved CSV without a GPU.
"""

import os

# Must be set before anything imports torch, otherwise it has no effect.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

from src.config.paths import DEFAULT_TRAIN_LLM_CONFIG_PATH, LLM_EVAL_OUTPUTS_DIR  # noqa: E402
from src.ml.config.loader import load_llm_training_config  # noqa: E402
from src.ml.llm.diagnostics import inspect_adapter, summarize  # noqa: E402
from src.utils.logging_config import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a fine-tuned adapter's raw outputs")
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAIN_LLM_CONFIG_PATH)
    parser.add_argument("--run-name", required=True, help="Training run name (Hub repo).")
    parser.add_argument("--split", help="Dataset split (default: config evaluation.split)")
    parser.add_argument("--size", type=int, default=200, help="Rows to inspect")
    parser.add_argument("--batch-size", type=int, help="Default: config evaluation.batch_size")
    parser.add_argument("--revision", help="Hub revision/branch of the adapter")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Skip the model; re-summarize the CSV a previous run wrote.",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")
    config = load_llm_training_config(args.config)
    split = args.split or config.evaluation.split
    csv_path = LLM_EVAL_OUTPUTS_DIR / args.run_name / f"inspection_{split}.csv"

    if args.summarize_only:
        summary = summarize(pd.read_csv(csv_path))
    else:
        frame, summary, tok_check = inspect_adapter(
            config,
            run_name=args.run_name,
            split=split,
            size=args.size,
            batch_size=args.batch_size or config.evaluation.batch_size,
            revision=args.revision,
        )
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(csv_path, index=False)
        tokenization = json.dumps(tok_check, indent=2, ensure_ascii=False)
        logger.info(f"Tokenization of the first row (as trl trains it):\n{tokenization}")
        logger.info(f"Per-game rows written to {csv_path}")

    logger.info(f"Summary:\n{json.dumps(summary, indent=2, default=float)}")


if __name__ == "__main__":
    main()
