"""Generate the gameId -> Polymarket slug lookup from the frozen holdout."""

from src.betting.game_slugs import make_game_slug_lookup
from src.utils.logging_config import setup_logging


def main() -> None:
    setup_logging(level="INFO")
    make_game_slug_lookup()


if __name__ == "__main__":
    main()
