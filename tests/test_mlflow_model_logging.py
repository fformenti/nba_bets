"""A trained model must actually be saved, and load back.

MLflow 3.17 changed the sklearn flavour's default format to skops, which refuses
xgboost and the calibration wrapper. The tracker caught that as a warning, so
from 2026-10-21 every training run reported a model URI with no model behind it
— and `--promote` would have deployed one.
"""

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.ml.tracking.mlflow_tracker import MLflowTracker


@pytest.fixture
def shipped_shape_pipeline():
    """The shape `train` ships: scaling, then calibrated xgboost."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(80, 3)), columns=["a", "b", "c"])
    y = (X["a"] + rng.normal(scale=0.5, size=80) > 0).astype(int)
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                CalibratedClassifierCV(
                    XGBClassifier(n_estimators=5, max_depth=2), method="sigmoid", cv=3
                ),
            ),
        ]
    )
    return pipeline.fit(X, y), X


def test_logged_model_loads_back(shipped_shape_pipeline, tmp_path):
    pipeline, X = shipped_shape_pipeline

    with MLflowTracker(
        experiment_name="logging_test", tracking_uri=f"sqlite:///{tmp_path}/mlflow.db"
    ) as tracker:
        tracker.log_model(pipeline, artifact_path="nba_xgboost", input_example=X.head(5))
        uri = f"runs:/{tracker.get_run_id()}/nba_xgboost"

    loaded = mlflow.sklearn.load_model(uri)
    np.testing.assert_allclose(loaded.predict_proba(X), pipeline.predict_proba(X))


def test_a_failed_save_is_not_swallowed(shipped_shape_pipeline, tmp_path, monkeypatch):
    pipeline, X = shipped_shape_pipeline

    def refuse(**kwargs):
        raise RuntimeError("untrusted types")

    monkeypatch.setattr(mlflow.sklearn, "log_model", refuse)

    with MLflowTracker(
        experiment_name="logging_test", tracking_uri=f"sqlite:///{tmp_path}/mlflow.db"
    ) as tracker:
        with pytest.raises(RuntimeError, match="untrusted types"):
            tracker.log_model(pipeline, artifact_path="nba_xgboost")
