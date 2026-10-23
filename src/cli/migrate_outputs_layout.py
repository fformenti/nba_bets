"""Move outputs/ into the run layout and convert the LLM evaluations. Run once.

See src/ml/results/migrate.py for what moves where, and docs/OUTPUTS.md for the
layout it produces. Pass --outputs-dir to migrate another copy, e.g. the
Drive mirror the Colab notebooks read.
"""

import argparse
import sys
from pathlib import Path

from src.config.paths import OUTPUTS_DIR
from src.ml.results.migrate import MigrationAborted, migrate_outputs_layout
from src.utils.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="One-time migration to the outputs/runs/ layout.")
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would move without moving or writing anything.",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")

    try:
        migrate_outputs_layout(args.outputs_dir, dry_run=args.dry_run)
    except MigrationAborted as exc:
        print(f"Migration aborted: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
