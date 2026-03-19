"""Baseline models for comparison with ML models."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)


class ThresholdBaseline(BaseEstimator, ClassifierMixin):
    """Base class for threshold-based baseline classifiers.

    Prediction rule: if feature_column >= 0 → predict 1, else 0.
    """

    def __init__(self, feature_column: str, name: str, name_caption: str):
        self.feature_column = feature_column
        self.is_fitted = False
        self.name = name
        self.name_caption = name_caption

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if self.feature_column not in X.columns:
            raise ValueError(
                f"Feature column '{self.feature_column}' not found in X. "
                f"Available columns: {list(X.columns)}"
            )
        self.is_fitted = True
        logger.info(f"Baseline model fitted using feature: {self.feature_column}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError(
                "Model must be fitted before prediction. Call fit() first."
            )
        if self.feature_column not in X.columns:
            raise ValueError(
                f"Feature column '{self.feature_column}' not found in X. "
                f"Available columns: {list(X.columns)}"
            )
        predictions = (X[self.feature_column] >= 0.0).astype(int).values
        logger.debug(
            f"Baseline predictions: {np.sum(predictions == 1)} wins, "
            f"{np.sum(predictions == 0)} losses"
        )
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        predictions = self.predict(X)
        proba = np.zeros((len(predictions), 2))
        proba[:, 0] = 1 - predictions
        proba[:, 1] = predictions
        return proba

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        from sklearn.metrics import accuracy_score

        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


class RecordDifferenceBaseline(ThresholdBaseline):
    """Baseline classifier based on record difference."""

    def __init__(self, feature_column: str = "record_L82_delta"):
        super().__init__(
            feature_column=feature_column,
            name="record_baseline",
            name_caption="Record Baseline",
        )


class PointDifferentialBaseline(ThresholdBaseline):
    """Baseline classifier based on point differential average delta."""

    def __init__(self, feature_column: str = "pts_diff_avg_L82_delta"):
        super().__init__(
            feature_column=feature_column,
            name="point_differential_baseline",
            name_caption="Point Differential Baseline",
        )


def create_baseline_model(
    feature_column: str = "pts_diff_avg_delta",
) -> PointDifferentialBaseline:
    """Create a baseline model instance."""
    return PointDifferentialBaseline(feature_column=feature_column)
