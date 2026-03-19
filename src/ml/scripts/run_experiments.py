"""Run multiple training experiments in sequence from config files."""

import argparse
import sys
from pathlib import Path

from src.ml.scripts.train_classifier import main as train_main
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


if __name__ == "__main__":
    setup_logging(level="INFO")

    parser = argparse.ArgumentParser(description="Run multiple training experiments")
    parser.add_argument("configs", nargs="+", type=Path, help="Config YAML files")
    args = parser.parse_args()

    failed = []
    for config_path in args.configs:
        logger.info(f"\n{'#' * 60}")
        logger.info(f"Running experiment: {config_path.stem}")
        logger.info(f"{'#' * 60}")
        try:
            train_main(config_path=config_path)
        except Exception as e:
            logger.error(f"Experiment {config_path.stem} failed: {e}", exc_info=True)
            failed.append(config_path.stem)

    if failed:
        logger.error(f"Failed experiments: {', '.join(failed)}")
        sys.exit(1)
    logger.info("All experiments completed successfully.")
