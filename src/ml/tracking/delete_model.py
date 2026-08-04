#!/usr/bin/env python3
"""
Script to delete MLflow registered models.

Usage:
    # Delete a specific model by name (and all its versions)
    uv run python -m src.ml.tracking.delete_model nba_classification_random_forest_all

    # Delete a specific version of a model
    uv run python -m src.ml.tracking.delete_model --version nba_classification_rf 3

    # Delete all model versions from an experiment
    uv run python -m src.ml.tracking.delete_model --experiment nba_bets_classification

    # Delete all registered models
    uv run python -m src.ml.tracking.delete_model --all
"""


import mlflow
from mlflow.exceptions import MlflowException

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def delete_model(model_name: str, tracking_uri: str = DEFAULT_TRACKING_URI):
    """Delete a specific registered model by name."""
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()

    try:
        model = client.get_registered_model(model_name)
        versions = model.latest_versions or []
        print(f"Found model: {model.name}")
        for v in versions:
            print(f"  Version {v.version}, Stage: {v.current_stage}, Run ID: {v.run_id}")

        response = input(
            f"\nAre you sure you want to delete model '{model_name}' "
            f"and all its versions? [y/N]: "
        )
        if response.lower() != "y":
            print("Deletion cancelled.")
            return

        client.delete_registered_model(model_name)
        print(f"Model '{model_name}' deleted successfully!")

    except MlflowException as e:
        if "not found" in str(e).lower():
            print(f"Model '{model_name}' does not exist.")
        else:
            print(f"Error: {e}")


def delete_model_version(
    model_name: str,
    version: str | int,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    *,
    confirm: bool = True,
):
    """Delete a specific version of a registered model."""
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()

    version_str = str(version)

    try:
        mv = client.get_model_version(model_name, version_str)
        print(f"Found model version: {model_name} (version {version_str})")
        print(f"  Stage: {mv.current_stage}, Run ID: {mv.run_id}")

        if confirm:
            response = input(
                f"\nAre you sure you want to delete {model_name} version {version_str}? [y/N]: "
            )
            if response.lower() != "y":
                print("Deletion cancelled.")
                return

        client.delete_model_version(model_name, version_str)
        print(f"Model '{model_name}' version {version_str} deleted successfully!")

    except MlflowException as e:
        if "not found" in str(e).lower():
            print(f"Model '{model_name}' version {version_str} does not exist.")
        else:
            print(f"Error: {e}")


def delete_models_by_experiment(
    experiment_name: str,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    *,
    confirm: bool = True,
):
    """Delete all registered model versions produced by runs in the given experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' does not exist.")
        return

    if experiment.lifecycle_stage == "deleted":
        print(f"Experiment '{experiment_name}' is already deleted.")
        return

    # Get all run IDs for this experiment
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    run_ids = {r.info.run_id for r in runs}

    if not run_ids:
        print(f"No runs found in experiment '{experiment_name}'.")
        return

    # Find all model versions produced by these runs
    versions_to_delete: list[tuple[str, str]] = []  # (model_name, version)
    seen: set[tuple[str, str]] = set()

    for run_id in run_ids:
        for mv in client.search_model_versions(filter_string=f"run_id = '{run_id}'"):
            key = (mv.name, mv.version)
            if key not in seen:
                seen.add(key)
                versions_to_delete.append(key)

    if not versions_to_delete:
        print(f"No registered model versions found for experiment '{experiment_name}'.")
        return

    print(f"Found {len(versions_to_delete)} model version(s) from experiment '{experiment_name}':")
    for model_name, version in versions_to_delete:
        print(f"  - {model_name} (version {version})")

    if confirm:
        response = input(
            f"\nAre you sure you want to delete these {len(versions_to_delete)} "
            f"model version(s)? [y/N]: "
        )
        if response.lower() != "y":
            print("Deletion cancelled.")
            return

    for model_name, version in versions_to_delete:
        client.delete_model_version(model_name, version)
        print(f"  Deleted {model_name} version {version}")

    # Delete any registered models that have no versions left
    model_names_touched = {name for name, _ in versions_to_delete}
    for model_name in model_names_touched:
        try:
            remaining = list(client.search_model_versions(filter_string=f"name = '{model_name}'"))
            if not remaining:
                client.delete_registered_model(model_name)
                print(f"  Deleted empty model '{model_name}'")
        except MlflowException:
            pass

    print(f"\nDeleted {len(versions_to_delete)} model version(s) from experiment '{experiment_name}'.")


def delete_all_models(tracking_uri: str = DEFAULT_TRACKING_URI):
    """Delete all registered models."""
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()

    models = client.search_registered_models()
    if not models:
        print("No registered models found.")
        return

    print("Found registered models:")
    for rm in models:
        print(f"  - {rm.name}")

    response = input(
        f"\nAre you sure you want to delete ALL {len(models)} registered models? [y/N]: "
    )
    if response.lower() != "y":
        print("Deletion cancelled.")
        return

    for rm in models:
        client.delete_registered_model(rm.name)
        print(f"  Deleted '{rm.name}'")

    print(f"\nAll {len(models)} models deleted successfully!")
