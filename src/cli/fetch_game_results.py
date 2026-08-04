"""Fetch final scores for games that have been played.

The provider is pluggable — see src/etl/collectors/results/. Use
``--source placeholder`` to read hand-dropped outcome files instead of a live
feed, which lets the whole daily cycle be exercised without one.
"""

import argparse
from pathlib import Path

from src.config.paths import UPCOMING_GAMES_DIR, UPCOMING_GAMES_RESULTS_DIR
from src.etl.collectors.results import DEFAULT_SOURCE, SOURCES
from src.etl.collectors.upcoming_games_results import enrich_upcoming_games_results
from src.utils.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich upcoming game JSON files with final scores."
    )
    parser.add_argument(
        "--source",
        choices=sorted(SOURCES),
        default=DEFAULT_SOURCE,
        help=f"Where outcomes come from (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument("--input-dir", type=Path, default=UPCOMING_GAMES_DIR)
    parser.add_argument("--output-dir", type=Path, default=UPCOMING_GAMES_RESULTS_DIR)
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Seconds between fetches. Defaults to the collector's rate limit; "
        "pass 0 for local sources that need no pacing.",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")

    kwargs = {} if args.delay is None else {"delay_seconds": args.delay}
    enrich_upcoming_games_results(
        args.input_dir, args.output_dir, source=args.source, **kwargs
    )


if __name__ == "__main__":
    main()
