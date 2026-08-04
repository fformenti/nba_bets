"""Freeze the holdout test set. Run once to fix the evaluation boundary."""

import argparse

from src.ml.datasets.holdout import (
    DEFAULT_HOLDOUT_START_SEASON,
    DEFAULT_MIN_SEASON,
    make_holdout_set,
)
from src.utils.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the holdout test set.")
    parser.add_argument(
        "--holdout-start-season",
        default=DEFAULT_HOLDOUT_START_SEASON,
        help=f"First season in the holdout (default: {DEFAULT_HOLDOUT_START_SEASON})",
    )
    parser.add_argument(
        "--min-season",
        default=DEFAULT_MIN_SEASON,
        help=f"Earliest season kept before splitting (default: {DEFAULT_MIN_SEASON})",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")
    make_holdout_set(args.holdout_start_season, args.min_season)


if __name__ == "__main__":
    main()
