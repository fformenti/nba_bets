"""SHAP compatibility helpers across SHAP and LightGBM versions."""

import numpy as np


def extract_binary_shap_values(explainer, X) -> np.ndarray:
    """Return SHAP values as (n_samples, n_features) for binary classification."""
    # Preferred API in newer SHAP versions.
    values = getattr(explainer(X), "values", None)
    if values is None:
        values = explainer.shap_values(X)

    # Older SHAP may return [neg_class, pos_class].
    if isinstance(values, list):
        return np.asarray(values[1])

    values = np.asarray(values)

    # Newer SHAP may return (n_samples, n_features, n_classes).
    if values.ndim == 3:
        return values[:, :, 1]

    return values
