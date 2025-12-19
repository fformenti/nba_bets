"""Evaluation metrics utilities."""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
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
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    # ROC AUC for binary classification
    if len(np.unique(y_true)) == 2 and y_pred_proba is not None:
        try:
            if y_pred_proba.ndim > 1:
                y_pred_proba = y_pred_proba[:, 1]
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
        except Exception:
            pass

    return metrics


def print_metrics_summary(metrics: Dict[str, float], prefix: str = ""):
    """
    Print a formatted summary of metrics.

    Parameters
    ----------
    metrics : dict
        Dictionary of metrics
    prefix : str, default=''
        Prefix to add to metric names
    """
    print(f"\n{'=' * 50}")
    print(f"Metrics Summary{' - ' + prefix if prefix else ''}")
    print(f"{'=' * 50}")

    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric_name:20s}: {value:10.4f}")
        else:
            print(f"{metric_name:20s}: {value}")

    print(f"{'=' * 50}\n")
