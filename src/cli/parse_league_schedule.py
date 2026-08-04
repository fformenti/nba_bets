"""Parse the league schedule CSV into its processed form."""

from src.etl.ingestion.parse_league_schedule import build_processed_schedule
from src.utils.logging_config import setup_logging


def main() -> None:
    setup_logging(level="INFO")
    build_processed_schedule()


if __name__ == "__main__":
    main()
