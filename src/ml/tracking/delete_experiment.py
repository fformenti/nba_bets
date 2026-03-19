#!/usr/bin/env python3
"""
Script to delete an MLflow experiment.

Usage:
    uv run python -m src.ml.tracking.delete_experiment <experiment_id_or_name>

Examples:
    uv run python -m src.ml.tracking.delete_experiment 1
    uv run python -m src.ml.tracking.delete_experiment my_experiment_name
"""

import sys
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def delete_experiment(experiment_id: str, tracking_uri: str = DEFAULT_TRACKING_URI):
    """Delete an MLflow experiment by ID."""
    mlflow.set_tracking_uri(tracking_uri)

    try:
        experiment = mlflow.get_experiment(experiment_id)
        print(f"Found experiment: {experiment.name} (ID: {experiment.experiment_id})")
        print(f"Artifact location: {experiment.artifact_location}")

        response = input(
            f"\nAre you sure you want to delete experiment "
            f"'{experiment.name}' (ID: {experiment_id})? [y/N]: "
        )
        if response.lower() != "y":
            print("Deletion cancelled.")
            return

        mlflow.delete_experiment(experiment_id)
        print(f"Experiment {experiment_id} ('{experiment.name}') deleted successfully!")

        artifact_dir = Path(experiment.artifact_location)
        if artifact_dir.exists():
            import shutil

            shutil.rmtree(artifact_dir)
            print(f"Removed artifact directory: {artifact_dir}")

    except MlflowException as e:
        if "does not exist" in str(e):
            print(f"Experiment {experiment_id} does not exist.")
        else:
            print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def delete_experiment_by_name(
    experiment_name: str, tracking_uri: str = DEFAULT_TRACKING_URI
):
    """Delete an MLflow experiment by name."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' does not exist.")
            return

        delete_experiment(experiment.experiment_id, tracking_uri)
    except MlflowException as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: uv run python -m src.ml.tracking.delete_experiment "
            "<experiment_id_or_name>"
        )
        sys.exit(1)

    arg = sys.argv[1]
    if arg.isdigit():
        delete_experiment(arg)
    else:
        delete_experiment_by_name(arg)
