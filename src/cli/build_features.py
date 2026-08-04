"""Build all feature tables and merge them into games_features.csv."""

import argparse

from src.config.paths import DEFAULT_FEATURES_CONFIG_PATH, project_relpath
from src.etl.make_features import build_features
from src.utils.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feature tables from game data")
    parser.add_argument(
        "--config",
        default=project_relpath(DEFAULT_FEATURES_CONFIG_PATH),
        help=f"Features config YAML (default: {project_relpath(DEFAULT_FEATURES_CONFIG_PATH)})",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")
    build_features(args.config)


if __name__ == "__main__":
    main()
