"""Build and save a Polymarket buying strategy for one game date."""

import argparse

from src.betting.bets import GAME_BUDGET, place_bets
from src.utils.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Place Polymarket bets for a game date")
    parser.add_argument("game_date", help="Game date to bet on, as YYYY-MM-DD")
    parser.add_argument(
        "--budget",
        type=float,
        default=GAME_BUDGET,
        help=f"Per-game budget in USDC (default: {GAME_BUDGET})",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")
    place_bets(args.game_date, budget=args.budget)


if __name__ == "__main__":
    main()
