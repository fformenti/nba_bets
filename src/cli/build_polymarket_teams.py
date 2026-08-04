"""Build the teamId -> Polymarket abbreviation lookup."""

from src.etl.reference.polymarket_teams import build_polymarket_teams_lookup
from src.utils.logging_config import setup_logging


def main() -> None:
    setup_logging(level="INFO")
    build_polymarket_teams_lookup()


if __name__ == "__main__":
    main()
