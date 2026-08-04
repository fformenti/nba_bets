"""Run predictions for upcoming games."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ml.prediction.pipeline import run_prediction_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate predictions for upcoming games."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to prediction config YAML file.",
    )
    args = parser.parse_args()

    run_prediction_pipeline(config_path=args.config)


if __name__ == "__main__":
    main()

