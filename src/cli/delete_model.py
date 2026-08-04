"""Delete MLflow registered models, versions, or everything."""

import argparse

from src.ml.tracking.delete_model import (
    delete_all_models,
    delete_model,
    delete_model_version,
    delete_models_by_experiment,
)
from src.utils.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete MLflow registered models")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--name", help="Delete this model and all its versions")
    group.add_argument(
        "--version", nargs=2, metavar=("NAME", "VERSION"), help="Delete one model version"
    )
    group.add_argument("--experiment", help="Delete all models from this experiment")
    group.add_argument("--all", action="store_true", help="Delete every registered model")
    args = parser.parse_args()

    setup_logging(level="INFO")
    if args.all:
        delete_all_models()
    elif args.experiment:
        delete_models_by_experiment(args.experiment)
    elif args.version:
        delete_model_version(*args.version)
    else:
        delete_model(args.name)


if __name__ == "__main__":
    main()
