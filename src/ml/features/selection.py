"""Boruta-SHAP feature selection.

Identifies truly informative features by comparing each feature's SHAP
importance against randomly-permuted "shadow" features across multiple
iterations, using a binomial test to confirm or reject each feature.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from scipy.stats import binomtest

from src.ml.config.schema import FeatureSelectionConfig
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class BorutaShapSelector:
    """Boruta-style feature selection using SHAP importances.

    Parameters
    ----------
    n_iterations : int
        Number of Boruta iterations.
    significance_level : float
        Significance level for the two-tailed binomial test.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_iterations: int = 20,
        significance_level: float = 0.05,
        random_state: int = 42,
    ):
        self.n_iterations = n_iterations
        self.significance_level = significance_level
        self.random_state = random_state

        self.confirmed_features_: list[str] = []
        self.rejected_features_: list[str] = []
        self.tentative_features_: list[str] = []
        self.feature_importances_: dict[str, float] = {}
        self.hit_counts_: dict[str, int] = {}
        self.max_shadow_importances_: list[float] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BorutaShapSelector":
        """Run the Boruta-SHAP algorithm on training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series
            Training target.

        Returns
        -------
        self
        """
        rng = np.random.RandomState(self.random_state)
        features = list(X.columns)
        n_features = len(features)

        hits = np.zeros(n_features, dtype=int)
        importance_accumulator = np.zeros(n_features, dtype=float)

        # Cast boolean columns to int for LightGBM compatibility
        X_work = X.copy()
        for col in X_work.columns:
            if X_work[col].dtype == bool:
                X_work[col] = X_work[col].astype(int)

        for iteration in range(1, self.n_iterations + 1):
            logger.info(f"Boruta-SHAP iteration {iteration}/{self.n_iterations}")

            # Create shadow features by shuffling each column independently
            X_shadow = X_work.apply(lambda col: col.sample(frac=1, random_state=rng.randint(0, 2**31)).values)
            X_shadow.columns = [f"shadow_{c}" for c in X_work.columns]

            X_extended = pd.concat([X_work, X_shadow], axis=1)

            estimator = LGBMClassifier(
                n_estimators=100,
                random_state=rng.randint(0, 2**31),
                verbose=-1,
                n_jobs=-1,
            )
            estimator.fit(X_extended, y)

            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_extended)

            # Handle binary classification output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            elif shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]

            mean_abs_shap = np.abs(shap_values).mean(axis=0)

            real_importances = mean_abs_shap[:n_features]
            shadow_importances = mean_abs_shap[n_features:]
            max_shadow = shadow_importances.max()
            self.max_shadow_importances_.append(float(max_shadow))

            importance_accumulator += real_importances

            for i in range(n_features):
                if real_importances[i] > max_shadow:
                    hits[i] += 1

        # Average importance across iterations
        avg_importances = importance_accumulator / self.n_iterations

        self.feature_importances_ = {f: float(avg_importances[i]) for i, f in enumerate(features)}
        self.hit_counts_ = {f: int(hits[i]) for i, f in enumerate(features)}

        # Two-tailed binomial test for each feature
        for i, feature in enumerate(features):
            result = binomtest(int(hits[i]), self.n_iterations, 0.5)
            p_value = result.pvalue

            if p_value < self.significance_level and hits[i] > self.n_iterations / 2:
                self.confirmed_features_.append(feature)
            elif p_value < self.significance_level and hits[i] <= self.n_iterations / 2:
                self.rejected_features_.append(feature)
            else:
                self.tentative_features_.append(feature)

        logger.info(
            f"Boruta-SHAP complete: {len(self.confirmed_features_)} confirmed, "
            f"{len(self.rejected_features_)} rejected, "
            f"{len(self.tentative_features_)} tentative"
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return only confirmed features."""
        return X[self.confirmed_features_]

    def get_support(self) -> dict[str, str]:
        """Return a mapping of feature name -> decision status."""
        support = {}
        for f in self.confirmed_features_:
            support[f] = "confirmed"
        for f in self.rejected_features_:
            support[f] = "rejected"
        for f in self.tentative_features_:
            support[f] = "tentative"
        return support

    def to_dict(self) -> dict:
        """Serialize results to a dictionary (for JSON artifact logging)."""
        return {
            "n_iterations": self.n_iterations,
            "significance_level": self.significance_level,
            "confirmed_features": self.confirmed_features_,
            "rejected_features": self.rejected_features_,
            "tentative_features": self.tentative_features_,
            "feature_importances": self.feature_importances_,
            "hit_counts": self.hit_counts_,
            "max_shadow_importances": self.max_shadow_importances_,
        }

    def save_json(self, path: Path) -> None:
        """Save results to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def run_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: FeatureSelectionConfig,
    random_state: int = 42,
) -> tuple[list[str], BorutaShapSelector]:
    """Run feature selection and return selected feature names.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    config : FeatureSelectionConfig
        Feature selection configuration.
    random_state : int
        Random seed.

    Returns
    -------
    tuple[list[str], BorutaShapSelector]
        (selected_feature_names, fitted_selector)
    """
    selector = BorutaShapSelector(
        n_iterations=config.n_iterations,
        significance_level=config.significance_level,
        random_state=config.random_state or random_state,
    )
    selector.fit(X_train, y_train)

    selected = list(selector.confirmed_features_)
    if config.include_tentative:
        selected += selector.tentative_features_

    if not selected:
        logger.warning(
            "Boruta-SHAP selected zero features. Falling back to all features."
        )
        selected = list(X_train.columns)

    logger.info(f"Selected {len(selected)} features: {selected}")
    return selected, selector
