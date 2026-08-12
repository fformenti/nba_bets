"""Promote the rebuilt history and drain the results inbox. Run once.

See src/etl/ingestion/migrate_incremental_layout.py for what this changes and
why it refuses to run when the inbox holds games the history does not.
"""

import argparse
import sys

from src.etl.ingestion.migrate_incremental_layout import (
    MigrationAborted,
    migrate_incremental_layout,
)
from src.utils.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-time migration to the staged incremental layout."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without moving files or rewriting history.",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")

    try:
        migrate_incremental_layout(dry_run=args.dry_run)
    except MigrationAborted as exc:
        print(f"Migration aborted: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
