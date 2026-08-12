"""Re-pull the league schedule so makeup dates for postponed games arrive."""

import argparse
from pathlib import Path

from src.config.paths import LEAGUE_SCHEDULE_PATH
from src.etl.collectors.league_schedule import fetch_league_schedule, season_string
from src.utils.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the league schedule from stats.nba.com."
    )
    parser.add_argument(
        "--season",
        default=None,
        help=f"Season in NBA form, e.g. {season_string()}. Defaults to the current one.",
    )
    parser.add_argument("--output", type=Path, default=LEAGUE_SCHEDULE_PATH)
    args = parser.parse_args()

    setup_logging(level="INFO")
    fetch_league_schedule(season=args.season, output_path=args.output)


if __name__ == "__main__":
    main()
