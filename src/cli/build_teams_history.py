"""Build the team/city/conference history lookup table."""

from src.etl.reference.teams_history import build_teams_history
from src.utils.logging_config import setup_logging


def main() -> None:
    setup_logging(level="INFO")
    build_teams_history()


if __name__ == "__main__":
    main()
