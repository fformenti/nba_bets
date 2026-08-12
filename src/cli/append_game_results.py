"""Merge fetched game results into the history table.

Consumed result files are moved to the archive, so the inbox always answers
"what has been fetched but not yet recorded?".
"""

import argparse
from pathlib import Path

from src.config.paths import (
    ARCHIVE_RESULTS_DIR,
    INGESTED_GAMES_UPDATED_HISTORY_PATH,
    UPCOMING_GAMES_RESULTS_DIR,
)
from src.etl.ingestion.append_games_results import add_game_results_to_historical
from src.utils.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append played-game results to the history table."
    )
    parser.add_argument("--results-dir", type=Path, default=UPCOMING_GAMES_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=INGESTED_GAMES_UPDATED_HISTORY_PATH)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=ARCHIVE_RESULTS_DIR,
        help="Where consumed result files are moved once recorded in history.",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")
    add_game_results_to_historical(
        upcoming_results_dir=args.results_dir,
        output_path=args.output,
        archive_dir=args.archive_dir,
    )


if __name__ == "__main__":
    main()
