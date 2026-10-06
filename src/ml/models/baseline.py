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
        logger.debug(f"Baseline model fitted using feature: {self.feature_column}")
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


class HomeWinBaseline(BaseEstimator, ClassifierMixin):
    """"The home team wins" — the floor any model has to clear.

    Not a :class:`ThresholdBaseline`: it reads no feature column at all, which is
    the point. Home advantage alone gets ~58% of NBA games right, so a model
    reporting 60% has bought four points, not sixty.

    The emitted probability is the home-win rate of the *training* split rather
    than a hard 1.0. Both give the same accuracy and the same ROC AUC (0.5 — a
    constant score cannot discriminate, which is the honest reading), but a hard
    1.0 would make this row's Brier and ECE incomparable to every other row in
    the table.
    """

    def __init__(self, home_win_rate: float = 0.5):
        self.home_win_rate = home_win_rate
        self.is_fitted = False
        self.name = "home_win_baseline"
        self.name_caption = "Home Team Wins Baseline"

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = np.empty((len(X), 2))
        proba[:, 1] = self.home_win_rate
        proba[:, 0] = 1.0 - self.home_win_rate
        return proba

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        from sklearn.metrics import accuracy_score

        return accuracy_score(y, self.predict(X))


class MarketProbabilityBaseline(BaseEstimator, ClassifierMixin):
    """Polymarket's tip-off price as a model — the ceiling worth beating.

    Takes a ``gameId → P(home win)`` mapping and looks each row up by its
    ``gameId`` column, so it is scored by the same ``ModelTrainer.evaluate`` path
    as every other row in the comparison table instead of by a parallel
    hand-rolled one.

    An unmapped gameId raises rather than imputing 0.5: the caller is expected to
    have already restricted the frame to games the market covers, and a silent
    coin-flip would enter the benchmark as an opinion the market never held.
    """

    def __init__(self, probabilities: pd.Series, id_column: str = "gameId"):
        self.probabilities = probabilities
        self.id_column = id_column
        self.is_fitted = False
        self.name = "polymarket_baseline"
        self.name_caption = "Polymarket Baseline"

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        self._lookup(X)
        self.is_fitted = True
        return self

    def _lookup(self, X: pd.DataFrame) -> np.ndarray:
        if self.id_column not in X.columns:
            raise ValueError(
                f"Column '{self.id_column}' not found in X. "
                f"Available columns: {list(X.columns)}"
            )
        ids = pd.to_numeric(X[self.id_column], errors="coerce").astype("Int64")
        probs = ids.map(self.probabilities)
        missing = int(probs.isna().sum())
        if missing:
            raise ValueError(
                f"{missing} of {len(X)} game(s) have no market price. Restrict the "
                "frame to mutually covered games before scoring."
            )
        return probs.astype(float).to_numpy()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self._lookup(X) >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self._lookup(X)
        proba = np.empty((len(X), 2))
        proba[:, 1] = p
        proba[:, 0] = 1.0 - p
        return proba

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        from sklearn.metrics import accuracy_score

        return accuracy_score(y, self.predict(X))
