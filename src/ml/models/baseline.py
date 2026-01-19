"""Baseline models for comparison with ML models."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)


class RecordDifferenceBaseline(BaseEstimator, ClassifierMixin):
    """
    Baseline classifier based on record difference.
    """

    def __init__(self, feature_column: str = "record_L82_delta"):
        """
        Initialize the baseline model.

        Parameters
        ----------
        feature_column : str, default='record_L82_delta'
            Name of the feature column to use for prediction
        """
        self.feature_column = feature_column
        self.is_fitted = False
        self.name = "best_record_baseline"
        self.name_caption = "Best Record Baseline"

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the baseline model (no-op, but required for sklearn interface).

        Parameters
        ----------
        X : pd.DataFrame
            Training features (must contain feature_column)
        y : pd.Series, optional
            Training target (not used, but required for sklearn interface)

        Returns
        -------
        self
            Returns self for method chaining
        """
        if self.feature_column not in X.columns:
            raise ValueError(
                f"Feature column '{self.feature_column}' not found in X. "
                f"Available columns: {list(X.columns)}"
            )

        self.is_fitted = True
        logger.info(f"Baseline model fitted using feature: {self.feature_column}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using the baseline rule.

        Parameters
        ----------
        X : pd.DataFrame
            Features (must contain feature_column)

        Returns
        -------
        np.ndarray
            Predicted class labels (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError(
                "Model must be fitted before prediction. Call fit() first."
            )

        if self.feature_column not in X.columns:
            raise ValueError(
                f"Feature column '{self.feature_column}' not found in X. "
                f"Available columns: {list(X.columns)}"
            )

        # Apply rule: record_difference >= 0 -> win_bool = 1, else 0
        predictions = (X[self.feature_column] >= 0.0).astype(int).values

        logger.debug(
            f"Baseline predictions: {np.sum(predictions == 1)} wins, "
            f"{np.sum(predictions == 0)} losses"
        )

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities using the baseline rule.

        For this baseline, probabilities are hard (1.0 or 0.0) based on the rule.

        Parameters
        ----------
        X : pd.DataFrame
            Features (must contain feature_column)

        Returns
        -------
        np.ndarray
            Predicted probabilities, shape (n_samples, 2)
            Column 0: probability of class 0 (loss)
            Column 1: probability of class 1 (win)
        """
        predictions = self.predict(X)

        # Convert to probabilities (hard probabilities: 1.0 or 0.0)
        proba = np.zeros((len(predictions), 2))
        proba[:, 0] = 1 - predictions  # Probability of class 0
        proba[:, 1] = predictions  # Probability of class 1

        return proba

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Calculate accuracy score.

        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series
            True labels

        Returns
        -------
        float
            Accuracy score
        """
        from sklearn.metrics import accuracy_score

        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


class PointDifferentialBaseline(BaseEstimator, ClassifierMixin):
    """
    Baseline classifier based on point differential average delta.

    Prediction rule:
    - If pts_diff_avg_L82_delta >= 0: predict win_bool = 1 (home team wins)
    - If pts_diff_avg_L82_delta < 0: predict win_bool = 0 (home team loses)

    This baseline uses the difference between home and away team's
    average point differential as a simple heuristic.
    """

    def __init__(self, feature_column: str = "pts_diff_avg_L82_delta"):
        """
        Initialize the baseline model.

        Parameters
        ----------
        feature_column : str, default='pts_diff_avg_delta'
            Name of the feature column to use for prediction
        """
        self.feature_column = feature_column
        self.is_fitted = False
        self.name = "point_differential_baseline"
        self.name_caption = "Point Differential Baseline"

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the baseline model (no-op, but required for sklearn interface).

        Parameters
        ----------
        X : pd.DataFrame
            Training features (must contain feature_column)
        y : pd.Series, optional
            Training target (not used, but required for sklearn interface)

        Returns
        -------
        self
            Returns self for method chaining
        """
        if self.feature_column not in X.columns:
            raise ValueError(
                f"Feature column '{self.feature_column}' not found in X. "
                f"Available columns: {list(X.columns)}"
            )

        self.is_fitted = True
        logger.info(f"Baseline model fitted using feature: {self.feature_column}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using the baseline rule.

        Parameters
        ----------
        X : pd.DataFrame
            Features (must contain feature_column)

        Returns
        -------
        np.ndarray
            Predicted class labels (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError(
                "Model must be fitted before prediction. Call fit() first."
            )

        if self.feature_column not in X.columns:
            raise ValueError(
                f"Feature column '{self.feature_column}' not found in X. "
                f"Available columns: {list(X.columns)}"
            )

        # Apply rule: pts_diff_avg_delta >= 0 -> win_bool = 1, else 0
        predictions = (X[self.feature_column] >= 0.0).astype(int).values

        logger.debug(
            f"Baseline predictions: {np.sum(predictions == 1)} wins, "
            f"{np.sum(predictions == 0)} losses"
        )

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities using the baseline rule.

        For this baseline, probabilities are hard (1.0 or 0.0) based on the rule.

        Parameters
        ----------
        X : pd.DataFrame
            Features (must contain feature_column)

        Returns
        -------
        np.ndarray
            Predicted probabilities, shape (n_samples, 2)
            Column 0: probability of class 0 (loss)
            Column 1: probability of class 1 (win)
        """
        predictions = self.predict(X)

        # Convert to probabilities (hard probabilities: 1.0 or 0.0)
        proba = np.zeros((len(predictions), 2))
        proba[:, 0] = 1 - predictions  # Probability of class 0
        proba[:, 1] = predictions  # Probability of class 1

        return proba

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Calculate accuracy score.

        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series
            True labels

        Returns
        -------
        float
            Accuracy score
        """
        from sklearn.metrics import accuracy_score

        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


def create_baseline_model(
    feature_column: str = "pts_diff_avg_delta",
) -> PointDifferentialBaseline:
    """
    Create a baseline model instance.

    Parameters
    ----------
    feature_column : str, default='pts_diff_avg_delta'
        Name of the feature column to use for prediction

    Returns
    -------
    PointDifferentialBaseline
        Baseline model instance
    """
    return PointDifferentialBaseline(feature_column=feature_column)
