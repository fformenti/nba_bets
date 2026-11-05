"""Put past games back in the pending queue, dated from the release schedule."""

import argparse

from src.etl.collectors.requeue_games import requeue_games
from src.utils.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write pending payloads for games history lacks or holds as postponed. "
        "Run `make fetch-game-results SOURCE=release` straight after."
    )
    parser.add_argument("game_ids", type=int, nargs="+", help="gameIds, e.g. 22400524")
    args = parser.parse_args()

    setup_logging(level="INFO")
    requeue_games(args.game_ids)


if __name__ == "__main__":
    main()
