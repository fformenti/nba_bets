"""Build the city-to-city distance table used by travel features."""

from src.etl.reference.distances import build_distances_table
from src.utils.logging_config import setup_logging


def main() -> None:
    setup_logging(level="INFO")
    build_distances_table()


if __name__ == "__main__":
    main()
