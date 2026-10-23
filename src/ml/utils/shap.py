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


def compute_shap_frame(
    model,
    X_transformed,
    feature_names: list,
    game_ids,
    max_samples: int = 500,
    random_state: int = 42,
):
    """SHAP values for a tree model, long: one row per (game_id, feature).

    Columns ``game_id``, ``feature``, ``shap_value``, ``feature_value`` - the
    shape of a run's ``model/shap.csv``. ``X_transformed`` has already been
    through the fitted preprocessor, so ``feature_value`` is in model space,
    which is what the beeswarm colours by. Rows are subsampled to
    ``max_samples`` for speed, as the beeswarm always was.
    """
    import pandas as pd
    import shap

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    X = np.asarray(X_transformed, dtype=np.float64)
    ids = np.asarray(game_ids)

    if X.shape[0] > max_samples:
        idx = np.random.RandomState(random_state).choice(X.shape[0], max_samples, replace=False)
        X, ids = X[idx], ids[idx]

    values = extract_binary_shap_values(shap.TreeExplainer(model), X)
    n_rows, n_features = values.shape
    return pd.DataFrame(
        {
            "game_id": np.repeat(ids, n_features),
            "feature": np.tile(np.asarray(feature_names, dtype=object), n_rows),
            "shap_value": values.ravel(),
            "feature_value": X.ravel(),
        }
    )
