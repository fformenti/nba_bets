"""Backfill Polymarket's tip-off odds per gameId, and keep them current."""

import argparse

from src.betting.polymarket_odds import DEFAULT_START_SEASON, build_polymarket_odds
from src.utils.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Polymarket tip-off odds for every game since a season"
    )
    parser.add_argument(
        "--start-season",
        default=DEFAULT_START_SEASON,
        help=f"First season to price, as YYYY/YY (default: {DEFAULT_START_SEASON})",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Retry games that previously resolved to no market",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")
    build_polymarket_odds(start_season=args.start_season, refresh=args.refresh)


if __name__ == "__main__":
    main()
