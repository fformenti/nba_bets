"""Release watched postponed games once the schedule gives them a makeup date."""

import argparse
from pathlib import Path

from src.config.paths import POSTPONED_GAMES_DIR, PROCESSED_LEAGUE_SCHEDULE_PATH
from src.etl.collectors.postponed_watch import reconcile_postponed_games
from src.utils.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check watched postponed games against the refreshed schedule."
    )
    parser.add_argument("--postponed-dir", type=Path, default=POSTPONED_GAMES_DIR)
    parser.add_argument("--schedule", type=Path, default=PROCESSED_LEAGUE_SCHEDULE_PATH)
    args = parser.parse_args()

    setup_logging(level="INFO")
    reconcile_postponed_games(
        postponed_dir=args.postponed_dir, schedule_path=args.schedule
    )


if __name__ == "__main__":
    main()
