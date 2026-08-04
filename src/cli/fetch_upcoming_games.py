"""Fetch the next slate of upcoming games from the processed league schedule."""

from src.etl.collectors.upcoming_games import get_upcoming_games
from src.utils.logging_config import setup_logging


def main() -> None:
    setup_logging(level="INFO")
    get_upcoming_games()


if __name__ == "__main__":
    main()
