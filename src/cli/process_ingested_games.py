"""Split ingested games into postponed and played regular-season sets."""

from src.etl.process_ingested_games import process_ingested_games
from src.utils.logging_config import setup_logging


def main() -> None:
    setup_logging(level="INFO")
    process_ingested_games()


if __name__ == "__main__":
    main()
