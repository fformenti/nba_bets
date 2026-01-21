#!/usr/bin/env python3
"""
Script to delete an MLflow experiment.

Usage:
    python delete_experiment.py <experiment_id>
    
Example:
    python delete_experiment.py 1
"""

import sys
import mlflow
from pathlib import Path

def delete_experiment(experiment_id: int):
    """Delete an MLflow experiment by ID."""
    # Set tracking URI to local
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    try:
        # Get experiment info
        experiment = mlflow.get_experiment(experiment_id)
        print(f"Found experiment: {experiment.name} (ID: {experiment.experiment_id})")
        print(f"Artifact location: {experiment.artifact_location}")
        
        # Confirm deletion
        response = input(f"\n⚠️  Are you sure you want to delete experiment '{experiment.name}' (ID: {experiment_id})? [y/N]: ")
        if response.lower() != 'y':
            print("Deletion cancelled.")
            return
        
        # Delete the experiment
        mlflow.delete_experiment(experiment_id)
        print(f"\n✅ Experiment {experiment_id} ('{experiment.name}') deleted successfully!")
        
        # Optionally remove the artifact directory if it still exists
        artifact_dir = Path(experiment.artifact_location)
        if artifact_dir.exists():
            import shutil
            shutil.rmtree(artifact_dir)
            print(f"✅ Removed artifact directory: {artifact_dir}")
        
    except mlflow.exceptions.MlflowException as e:
        if "does not exist" in str(e):
            print(f"❌ Experiment {experiment_id} does not exist.")
        else:
            print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python delete_experiment.py <experiment_id>")
        sys.exit(1)
    
    try:
        experiment_id = int(sys.argv[1])
        delete_experiment(experiment_id)
    except ValueError:
        print("Error: Experiment ID must be an integer")
        sys.exit(1)

