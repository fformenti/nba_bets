"""Put several runs - sklearn, regressor or LLM - on the same games and axes.

    make plot-compare NAME=llm_vs_xgb SERIES="e11-llama32-1b-compact xgboost-20261023-141500"
    make plot-compare NAME=calibrators SPLIT=test \\
        SERIES="xgboost-20261023-141500:xgboost:none xgboost-20261023-141500:xgboost:isotonic"

Each series is ``run_id[:model[:calibration]]``; omitted parts default to what
the run's run.json selected. Writes outputs/comparisons/<NAME>/. See
docs/OUTPUTS.md.
"""

import argparse

import matplotlib

# Files only, never a window.
matplotlib.use("Agg")

from src.ml.plotting.report import build_comparison  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare runs on their shared games")
    parser.add_argument("--series", nargs="+", required=True, help="run_id[:model[:calibration]]")
    parser.add_argument("--name", required=True, help="Folder name under outputs/comparisons/")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument(
        "--market",
        action="store_true",
        help="Add Polymarket as a series (narrows to the games it priced: test seasons).",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Add the record baseline: the better season-to-date record wins, ties go home.",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")
    build_comparison(
        args.series, args.name, split=args.split,
        include_market=args.market, include_record=args.record,
    )


if __name__ == "__main__":
    main()
