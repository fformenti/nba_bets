"""Dump a fine-tuned adapter's raw answers and first-token sign probabilities.

    make inspect-llm LLM_RUN=nba-bets-2026-07-27

Writes ``outputs/runs/<run_name>/llm/inspection_<split>.csv`` - one row per
game with the raw decoded text, the parsed number, P(+), P(-), the expected
margin and the top first tokens - plus the run's shared result files
(``predictions/<split>.csv``, ``metrics/<split>.json``, ``run.json``) that
``make plot`` and ``make plot-compare`` read. ``--summarize-only`` re-summarizes
a saved CSV without a GPU.

Baselines for the untuned base model (``--run-name`` then only names the
output folder)::

    make inspect-llm LLM_RUN=base-8b ARGS="--base-only --split validation"
    make inspect-llm LLM_RUN=base-8b-5shot ARGS="--base-only --few-shot 5 --split validation"

Platt recalibration, CPU only, from the validation inspection of the same run::

    make inspect-llm LLM_RUN=test_run2 ARGS="--summarize-only --split test --calibrate-with validation"
"""

import os

# Must be set before anything imports torch, otherwise it has no effect.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402


from src.config.paths import DEFAULT_TRAIN_LLM_CONFIG_PATH, RunPaths  # noqa: E402
from src.ml.config.loader import load_llm_training_config  # noqa: E402
from src.ml.llm.diagnostics import (  # noqa: E402
    inspect_adapter,
    read_inspection_csv,
    recalibrate,
    summarize,
)
from src.ml.llm.finetune import build_hub_model_id  # noqa: E402
from src.ml.llm.results import write_inspection_results  # noqa: E402
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
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Score the untuned base model; --run-name only names the output folder.",
    )
    parser.add_argument(
        "--few-shot",
        type=int,
        default=0,
        help="Prepend this many solved training examples to every prompt (baselines).",
    )
    parser.add_argument(
        "--calibrate-with",
        metavar="SPLIT",
        help="With --summarize-only: also summarize after Platt recalibration fitted "
        "on this split's saved CSV (cross-fitted when it is --split itself).",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")
    config = load_llm_training_config(args.config)
    split = args.split or config.evaluation.split
    run_paths = RunPaths(args.run_name)
    csv_path = run_paths.inspection(split)

    if args.summarize_only:
        frame = read_inspection_csv(csv_path)
        summary = summarize(frame)
    else:
        frame, summary, tok_check = inspect_adapter(
            config,
            run_name=args.run_name,
            split=split,
            size=args.size,
            batch_size=args.batch_size or config.evaluation.batch_size,
            revision=args.revision,
            base_only=args.base_only,
            few_shot=args.few_shot,
        )
        write_inspection_results(
            frame,
            run_id=args.run_name,
            split=split,
            seed=config.evaluation.seed,
            config_path=str(args.config),
            extra={
                "base_model": config.base_model,
                "hub_model_id": None if args.base_only else build_hub_model_id(config, args.run_name),
                "revision": args.revision,
                "base_only": args.base_only,
                "few_shot": args.few_shot,
            },
        )
        tokenization = json.dumps(tok_check, indent=2, ensure_ascii=False)
        logger.info(f"Tokenization of the first row (as trl trains it):\n{tokenization}")
        logger.info(f"Per-game rows written to {csv_path}")

    logger.info(f"Summary:\n{json.dumps(summary, indent=2, default=float)}")

    if args.calibrate_with:
        calibration = (
            None
            if args.calibrate_with == split
            else read_inspection_csv(run_paths.inspection(args.calibrate_with))
        )
        calibrated = summarize(recalibrate(frame, calibration, seed=config.evaluation.seed))
        logger.info(
            f"Summary after Platt recalibration (fit on {args.calibrate_with}):\n"
            f"{json.dumps(calibrated, indent=2, default=float)}"
        )


if __name__ == "__main__":
    main()
