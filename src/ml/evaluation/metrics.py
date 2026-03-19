"""Evaluation metrics utilities."""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
)


def compute_regression_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute regression metrics.

    Parameters
    ----------
    y_true : pd.Series
        True target values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    dict
        Dictionary of metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
    }


def compute_brier_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Compute Brier score (lower is better, 0 = perfect calibration)."""
    return float(brier_score_loss(y_true, y_pred_proba))


def compute_ece(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE).

    Bins predictions into equal-width bins and computes the weighted average
    of |mean(y_true) - mean(y_pred_proba)| per bin.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)
    for i in range(n_bins):
        mask = (y_pred_proba > bin_edges[i]) & (y_pred_proba <= bin_edges[i + 1])
        if i == 0:
            mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba <= bin_edges[i + 1])
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_pred = y_pred_proba[mask].mean()
        avg_true = np.array(y_true)[mask].mean()
        ece += (n_bin / n_total) * abs(avg_true - avg_pred)
    return float(ece)


def compute_classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    average: str = "weighted",
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : pd.Series
        True target values
    y_pred : np.ndarray
        Predicted class labels
    y_pred_proba : np.ndarray, optional
        Predicted probabilities (for ROC AUC)
    average : str, default='weighted'
        Averaging strategy for multi-class metrics

    Returns
    -------
    dict
        Dictionary of metrics
    """
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
    }

    # ROC AUC, Brier score, and ECE for binary classification
    if len(np.unique(y_true)) == 2 and y_pred_proba is not None:
        try:
            if y_pred_proba.ndim > 1:
                y_pred_proba = y_pred_proba[:, 1]
            metrics["roc_auc"] = round(roc_auc_score(y_true, y_pred_proba), 4)
            metrics["brier_score"] = round(compute_brier_score(y_true, y_pred_proba), 4)
            metrics["ece"] = round(compute_ece(y_true, y_pred_proba), 4)
        except Exception:
            pass

    return metrics


MODEL_DISPLAY_NAMES = {
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
    "lgbm": "LightGBM",
    "best_record_baseline": "Best Record Baseline",
    "point_differential_baseline": "Point Differential Baseline",
}

CONFERENCE_DISPLAY_NAMES = {
    "same": "Same Conference",
    "different": "Different Conferences",
    "all": "All Games",
}


def get_model_display_name(model_key: str) -> str:
    """Return a human-readable display name for a model key."""
    return MODEL_DISPLAY_NAMES.get(model_key, model_key)


def get_conference_display_name(conference_filter: str) -> str:
    """Return a human-readable display name for a conference filter."""
    return CONFERENCE_DISPLAY_NAMES.get(conference_filter, conference_filter)


def format_metrics_line(metrics: Dict[str, float], prefix: str = "") -> str:
    """
    Format key metrics (accuracy and ROC AUC) as a single-line string with percentages.

    Parameters
    ----------
    metrics : dict
        Dictionary of metrics (keys may be prefixed, e.g. 'test_accuracy')
    prefix : str, default=''
        Expected prefix in metric keys (e.g. 'test', 'val', 'train')

    Returns
    -------
    str
        Formatted string like "Accuracy: 65.23% | ROC AUC: 71.50%"
    """
    acc_key = f"{prefix}_accuracy" if prefix else "accuracy"
    auc_key = f"{prefix}_roc_auc" if prefix else "roc_auc"

    parts = []
    if acc_key in metrics:
        parts.append(f"Accuracy: {metrics[acc_key] * 100:.2f}%")
    if auc_key in metrics:
        parts.append(f"ROC AUC: {metrics[auc_key] * 100:.2f}%")

    return " | ".join(parts)


def print_metrics_summary(
    metrics: Dict[str, float],
    model_name: str = "",
    split: str = "",
    conference_filter: str = "",
) -> None:
    """
    Print a formatted summary of key metrics (accuracy and ROC AUC) as percentages.

    Parameters
    ----------
    metrics : dict
        Dictionary of metrics (keys may be prefixed, e.g. 'test_accuracy')
    model_name : str, default=''
        Model key name (e.g. 'random_forest')
    split : str, default=''
        Data split prefix in metric keys (e.g. 'test', 'val', 'train')
    conference_filter : str, default=''
        Conference filter key (e.g. 'same', 'different', 'all')
    """
    line = format_metrics_line(metrics, prefix=split)
    if not line:
        return

    display_name = get_model_display_name(model_name) if model_name else "Model"
    conf_label = f" ({get_conference_display_name(conference_filter)})" if conference_filter else ""
    split_label = f" [{split.capitalize()}]" if split else ""
    print(f"  {display_name}{conf_label}{split_label}: {line}")
