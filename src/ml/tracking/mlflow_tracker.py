"""
MLflow integration for experiment tracking.

This module provides utilities to track ML experiments using MLflow,
including parameters, metrics, models, and artifacts.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Avoid recording env vars in MLflow model logging unless explicitly enabled.
os.environ.setdefault("MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING", "false")

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def setup_mlflow_experiment(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    artifact_location: Optional[str] = None,
) -> str:
    """
    Set up an MLflow experiment.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment
    tracking_uri : str, optional
        MLflow tracking URI. If None, uses default local file system.
    artifact_location : str, optional
        Artifact storage location. If None, uses default.

    Returns
    -------
    str
        Experiment ID
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name, artifact_location=artifact_location
            )
            logger.info(
                f"Created new MLflow experiment: {experiment_name} (ID: {experiment_id})"
            )
        else:
            # Check if experiment is deleted
            if experiment.lifecycle_stage == "deleted":
                logger.info(
                    f"Found deleted experiment: {experiment_name}. Restoring it."
                )
                client = MlflowClient()
                client.restore_experiment(experiment.experiment_id)
            experiment_id = experiment.experiment_id
            logger.info(
                f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})"
            )
    except Exception as e:
        logger.warning(
            f"Error setting up MLflow experiment: {e}. Using default experiment."
        )
        experiment_id = mlflow.get_experiment_by_name("Default").experiment_id

    mlflow.set_experiment(experiment_name)
    return experiment_id


class MLflowTracker:
    """
    Context manager for MLflow experiment tracking.

    This class provides a convenient way to track ML experiments with automatic
    logging of parameters, metrics, models, and artifacts.
    """

    def __init__(
        self,
        experiment_name: str = "nba_bets",
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        log_model: bool = True,
    ):
        """
        Initialize MLflow tracker.

        Parameters
        ----------
        experiment_name : str, default='nba_bets'
            Name of the MLflow experiment
        run_name : str, optional
            Name for this specific run. If None, MLflow will auto-generate.
        tracking_uri : str, optional
            MLflow tracking URI. If None, uses default local file system.
        log_model : bool, default=True
            Whether to log models to MLflow
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self._log_model_enabled = log_model
        self.active_run = None

    def __enter__(self):
        """Start MLflow run."""
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        setup_mlflow_experiment(self.experiment_name)
        self.active_run = mlflow.start_run(run_name=self.run_name)
        logger.info(f"Started MLflow run: {self.active_run.info.run_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End MLflow run."""
        if self.active_run:
            mlflow.end_run()
            logger.info(f"Ended MLflow run: {self.active_run.info.run_id}")

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow.

        Parameters
        ----------
        params : dict
            Dictionary of parameters to log
        """
        mlflow.log_params(params)
        logger.debug(f"Logged {len(params)} parameters to MLflow")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to MLflow.

        Parameters
        ----------
        metrics : dict
            Dictionary of metrics to log
        step : int, optional
            Step number for metrics (useful for iterative training)
        """
        mlflow.log_metrics(metrics, step=step)
        logger.debug(f"Logged {len(metrics)} metrics to MLflow")

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
        input_example: Optional[Any] = None,
    ):
        """
        Log a model to MLflow.

        Parameters
        ----------
        model : Any
            Model to log (scikit-learn model or pipeline)
        artifact_path : str, default='model'
            Artifact name within the run's artifact directory
        registered_model_name : str, optional
            If provided, register the model with this name in the model registry
        input_example : Any, optional
            Example inputs used to infer model signature and log input example
        """
        if not self._log_model_enabled:
            return

        extra_pip_requirements = None
        try:
            import pip  # type: ignore

            extra_pip_requirements = [f"pip=={pip.__version__}"]
        except Exception:
            extra_pip_requirements = None

        signature = None
        if input_example is not None:
            try:
                model_output = model.predict(input_example)
                signature = infer_signature(input_example, model_output)
            except Exception as e:
                logger.warning(f"Could not infer MLflow signature: {e}")

        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                name=artifact_path,
                registered_model_name=registered_model_name,
                extra_pip_requirements=extra_pip_requirements,
                signature=signature,
                input_example=input_example,
            )
            logger.info(f"Logged model to MLflow at {artifact_path}")
        except Exception as e:
            logger.warning(f"Error logging model to MLflow: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact (file) to MLflow.

        Parameters
        ----------
        local_path : str
            Path to local file to log
        artifact_path : str, optional
            Path within the run's artifact directory. If None, uses filename.
        """
        mlflow.log_artifact(local_path, artifact_path=artifact_path)
        logger.debug(f"Logged artifact: {local_path}")

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """
        Log all files in a directory as artifacts.

        Parameters
        ----------
        local_dir : str
            Path to local directory to log
        artifact_path : str, optional
            Path within the run's artifact directory
        """
        mlflow.log_artifacts(local_dir, artifact_path=artifact_path)
        logger.debug(f"Logged artifacts from directory: {local_dir}")

    def log_config(self, config: Dict[str, Any], config_path: str = "config.yaml"):
        """
        Log configuration as an artifact.

        Parameters
        ----------
        config : dict
            Configuration dictionary
        config_path : str, default='config.yaml'
            Path to save config within artifacts
        """
        import yaml
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f, default_flow_style=False)
            temp_path = f.name

        try:
            self.log_artifact(temp_path, artifact_path=config_path)
        finally:
            Path(temp_path).unlink()

    def set_tags(self, tags: Dict[str, str]):
        """
        Set tags for the current run.

        Parameters
        ----------
        tags : dict
            Dictionary of tag key-value pairs
        """
        mlflow.set_tags(tags)
        logger.debug(f"Set {len(tags)} tags on MLflow run")

    def get_run_id(self) -> Optional[str]:
        """
        Get the current run ID.

        Returns
        -------
        str, optional
            Run ID if a run is active, None otherwise
        """
        if self.active_run:
            return self.active_run.info.run_id
        return None
