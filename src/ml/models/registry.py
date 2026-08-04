"""Model persistence and registry utilities."""

import joblib
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from sklearn.base import BaseEstimator


class ModelRegistry:
    """Registry for managing saved models and their metadata."""

    def __init__(self, registry_path: Path):
        """
        Initialize ModelRegistry.

        Parameters
        ----------
        registry_path : Path
            Path to the model registry directory
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_path / "metadata.json"
        self._load_metadata()

    def _load_metadata(self):
        """Load metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def save(
        self,
        model: BaseEstimator,
        model_name: str,
        task_type: str,
        metrics: Optional[Dict[str, Any]] = None,
        feature_names: Optional[list] = None,
        **kwargs,
    ) -> Path:
        """
        Save a model to the registry.

        Parameters
        ----------
        model : BaseEstimator
            Trained model to save
        model_name : str
            Name identifier for the model
        task_type : str
            Type of task ('regression' or 'classification')
        metrics : dict, optional
            Model performance metrics
        feature_names : list, optional
            List of feature names used by the model
        **kwargs
            Additional metadata to store

        Returns
        -------
        Path
            Path to saved model file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.joblib"
        model_path = self.registry_path / model_filename

        # Save model
        joblib.dump(model, model_path)

        # Save metadata
        model_metadata = {
            "model_name": model_name,
            "task_type": task_type,
            "timestamp": timestamp,
            "model_path": str(model_path.relative_to(self.registry_path)),
            "metrics": metrics or {},
            "feature_names": feature_names,
            **kwargs,
        }

        if model_name not in self.metadata:
            self.metadata[model_name] = []

        self.metadata[model_name].append(model_metadata)
        self._save_metadata()

        return model_path

    def load(
        self, model_name: str, version: Optional[int] = None
    ) -> tuple[BaseEstimator, Dict[str, Any]]:
        """
        Load a model from the registry.

        Parameters
        ----------
        model_name : str
            Name identifier for the model
        version : int, optional
            Version index (0 = latest, -1 = oldest). If None, loads latest.

        Returns
        -------
        tuple
            Model and metadata
        """
        if model_name not in self.metadata:
            raise ValueError(f"Model '{model_name}' not found in registry")

        versions = self.metadata[model_name]

        if version is None:
            version = -1  # Latest

        model_info = versions[version]
        model_path = self.registry_path / model_info["model_path"]

        model = joblib.load(model_path)

        return model, model_info

    def list_models(self) -> list[str]:
        """List all model names in the registry."""
        return list(self.metadata.keys())

    def get_model_versions(self, model_name: str) -> list[Dict[str, Any]]:
        """Get all versions of a model."""
        if model_name not in self.metadata:
            return []
        return self.metadata[model_name]


def load_model(file_path: Path) -> BaseEstimator:
    """
    Load a model from disk.

    Parameters
    ----------
    file_path : Path
        Path to the model file

    Returns
    -------
    BaseEstimator
        Loaded model
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")

    return joblib.load(file_path)
