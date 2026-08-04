"""Predict outcomes for the fetched upcoming games."""

import argparse
from pathlib import Path

from src.ml.prediction.pipeline import run_prediction_pipeline
from src.utils.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict outcomes for upcoming NBA games")
    parser.add_argument("--config", type=Path, help="Path to prediction config YAML")
    args = parser.parse_args()

    setup_logging(level="INFO")
    run_prediction_pipeline(config_path=args.config)


if __name__ == "__main__":
    main()
