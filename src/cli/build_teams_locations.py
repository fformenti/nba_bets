"""Look up each team's city and state and persist the locations table."""

from src.etl.reference.teams_locations import build_teams_locations
from src.utils.logging_config import setup_logging


def main() -> None:
    setup_logging(level="INFO")
    build_teams_locations()


if __name__ == "__main__":
    main()
