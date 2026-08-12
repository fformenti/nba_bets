"""Train the classification models for one experiment config."""

import argparse
from pathlib import Path

from src.ml.training.classifier import train_classifier
from src.utils.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Train classification model")
    parser.add_argument("--config", type=Path, help="Path to training config YAML")
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Deploy the resulting model: point configs/predict/predict_classifier.yaml "
        "at this run. Off by default — training and deploying are separate decisions.",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")
    train_classifier(config_path=args.config, promote=args.promote)


if __name__ == "__main__":
    main()
