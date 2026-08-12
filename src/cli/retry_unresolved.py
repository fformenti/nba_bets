"""Let quarantined games back into the pipeline after investigating them."""

import argparse
from pathlib import Path

from src.config.paths import UNRESOLVED_GAMES_DIR
from src.etl.collectors.postponed_watch import retry_unresolved_games
from src.utils.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clear the quarantine so those games are collected again."
    )
    parser.add_argument("--unresolved-dir", type=Path, default=UNRESOLVED_GAMES_DIR)
    args = parser.parse_args()

    setup_logging(level="INFO")
    retry_unresolved_games(unresolved_dir=args.unresolved_dir)


if __name__ == "__main__":
    main()
