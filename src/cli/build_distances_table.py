"""Build the city-to-city distance table used by travel features.

Calls Serper and OpenAI once per city pair. The result is a reference *input*
under data/raw/reference/, not a build artifact — see docs/PIPELINE_AUDIT.md.
"""

import argparse

from src.etl.reference.distances import build_distances_table
from src.utils.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the city-to-city distance reference table."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many city pairs. Smoke-testing the API wiring only "
        "— a partial table makes every unmatched trip read as zero miles.",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")
    build_distances_table(limit=args.limit)


if __name__ == "__main__":
    main()
