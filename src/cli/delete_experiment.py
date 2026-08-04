"""Delete an MLflow experiment by id or name."""

import argparse

from src.ml.tracking.delete_experiment import delete_experiment, delete_experiment_by_name
from src.utils.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete an MLflow experiment")
    parser.add_argument("experiment", help="Experiment id or name")
    args = parser.parse_args()

    setup_logging(level="INFO")
    if args.experiment.isdigit():
        delete_experiment(args.experiment)
    else:
        delete_experiment_by_name(args.experiment)


if __name__ == "__main__":
    main()
