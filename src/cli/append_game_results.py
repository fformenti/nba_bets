"""Merge fetched game results into the historical ingested games."""

import argparse

from src.etl.ingestion.append_games_results import add_game_results_to_historical
from src.utils.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append played-game results to the historical set."
    )
    parser.add_argument(
        "--keep-old-ids",
        action="store_true",
        help="Retain pre-existing gameIds instead of renumbering.",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")
    add_game_results_to_historical(keep_old_ids=args.keep_old_ids)


if __name__ == "__main__":
    main()
