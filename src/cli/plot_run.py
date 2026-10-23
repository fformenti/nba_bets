"""Draw one run's figures from its result files.

    make plot RUN=xgboost-20261023-141500
    make plot RUN=latest              # the most recent run of any family
    make plot RUN=e4-llama32-1b SPLIT=validation

Writes outputs/runs/<run_id>/report/{figures,tables}/ and, when the run has an
MLflow run, attaches that folder to it. See docs/OUTPUTS.md.
"""

import argparse

import matplotlib

# Files only, never a window: this runs headless after `make train`.
matplotlib.use("Agg")

from src.ml.plotting.report import build_run_report, log_report_to_mlflow  # noqa: E402
from src.ml.results import latest_run_id  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a run's report from its result files")
    parser.add_argument("--run", required=True, help="Run id under outputs/runs/, or 'latest'")
    parser.add_argument("--split", choices=["validation", "test"], help="Default: test if scored")
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Do not attach the report to the MLflow run."
    )
    args = parser.parse_args()

    setup_logging(level="INFO")
    run_id = latest_run_id() if args.run == "latest" else args.run
    build_run_report(run_id, split=args.split)
    if not args.no_mlflow:
        log_report_to_mlflow(run_id)


if __name__ == "__main__":
    main()
