"""Run several training experiments in sequence."""

import argparse
import sys
from pathlib import Path

from src.ml.training.classifier import train_classifier
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple training experiments")
    parser.add_argument("configs", nargs="+", type=Path, help="Config YAML files")
    args = parser.parse_args()

    setup_logging(level="INFO")

    failed = []
    for config_path in args.configs:
        logger.info(f"\n{'#' * 60}")
        logger.info(f"Running experiment: {config_path.stem}")
        logger.info(f"{'#' * 60}")
        try:
            train_classifier(config_path=config_path)
        except Exception as exc:
            # Keep going: one bad config should not cost the whole batch.
            logger.error(f"Experiment {config_path.stem} failed: {exc}", exc_info=True)
            failed.append(config_path.stem)

    if failed:
        logger.error(f"Failed experiments: {', '.join(failed)}")
        sys.exit(1)
    logger.info("All experiments completed successfully.")


if __name__ == "__main__":
    main()
