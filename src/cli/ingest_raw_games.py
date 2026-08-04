"""Parse data/raw/historical/games/Games.csv into the ingested format."""

from src.etl.ingestion.raw_games import ingest_raw_games
from src.utils.logging_config import setup_logging


def main() -> None:
    setup_logging(level="INFO")
    ingest_raw_games()


if __name__ == "__main__":
    main()
