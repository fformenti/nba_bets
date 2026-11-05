"""Download team and player box scores, then validate what is stored."""

import argparse

from src.config.paths import BOXSCORES_VALIDATION_PATH
from src.etl.collectors.boxscores import fetch_boxscores
from src.etl.collectors.boxscores_validation import load_history_games, validate_boxscores
from src.etl.collectors.sportsdataverse import all_seasons
from src.etl.utils.common import atomic_write_csv
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill box scores from the sportsdataverse releases and refresh "
        "the current season when its release has moved."
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=None,
        help="Season END years, e.g. 2025 for 2024/25. Defaults to 1997 through the current one.",
    )
    parser.add_argument(
        "--refresh-all",
        action="store_true",
        help="Re-download every season whose release timestamp changed, not just the current one.",
    )
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    setup_logging(level="INFO")
    seasons = args.seasons or all_seasons()
    outcomes = fetch_boxscores(seasons, refresh_all=args.refresh_all)
    downloaded = sum(outcome.action == "downloaded" for outcome in outcomes)
    logger.info(f"Downloaded {downloaded} files, skipped {len(outcomes) - downloaded}")

    if args.no_validate:
        return
    report = validate_boxscores(seasons, load_history_games())
    atomic_write_csv(report, BOXSCORES_VALIDATION_PATH)
    logger.info(
        f"Validation written to {BOXSCORES_VALIDATION_PATH}\n{report.to_string(index=False)}"
    )


if __name__ == "__main__":
    main()
