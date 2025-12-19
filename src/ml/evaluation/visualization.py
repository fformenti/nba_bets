"""Visualization utilities for model evaluation."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def plot_residuals(
    y_true: pd.Series,
    y_pred: np.ndarray,
    title: str = "Residual Plot",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot residuals for regression models.

    Parameters
    ----------
    y_true : pd.Series
        True target values
    y_pred : np.ndarray
        Predicted values
    title : str, default='Residual Plot'
        Plot title
    figsize : tuple, default=(10, 6)
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color="r", linestyle="--")
    axes[0].set_xlabel("Predicted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Predicted")
    axes[0].grid(True, alpha=0.3)

    # Residuals distribution
    axes[1].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    axes[1].axvline(x=0, color="r", linestyle="--")
    axes[1].set_xlabel("Residuals")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residuals Distribution")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    return fig


def plot_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    title: str = "Predictions vs Actual",
    figsize: tuple = (8, 8),
) -> plt.Figure:
    """
    Plot predictions against actual values.

    Parameters
    ----------
    y_true : pd.Series
        True target values
    y_pred : np.ndarray
        Predicted values
    title : str, default='Predictions vs Actual'
        Plot title
    figsize : tuple, default=(8, 8)
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.plot(
        [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction"
    )
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
    title: str = "Confusion Matrix",
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """
    Plot confusion matrix for classification models.

    Parameters
    ----------
    y_true : pd.Series
        True class labels
    y_pred : np.ndarray
        Predicted class labels
    class_names : list, optional
        Names of classes
    title : str, default='Confusion Matrix'
        Plot title
    figsize : tuple, default=(8, 6)
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    plt.tight_layout()

    return fig


def plot_feature_importance(
    model: BaseEstimator,
    feature_names: list,
    top_n: int = 20,
    title: str = "Feature Importance",
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    Plot feature importance for tree-based models.

    Parameters
    ----------
    model : BaseEstimator
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    top_n : int, default=20
        Number of top features to display
    title : str, default='Feature Importance'
        Plot title
    figsize : tuple, default=(10, 8)
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not have feature_importances_ attribute")

    importances = model.feature_importances_

    # Create DataFrame for easier sorting
    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        )
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(range(len(importance_df)), importance_df["importance"])
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df["feature"])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    return fig


def plot_learning_curves(
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    train_sizes: np.ndarray,
    title: str = "Learning Curves",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot learning curves.

    Parameters
    ----------
    train_scores : np.ndarray
        Training scores for different training set sizes
    val_scores : np.ndarray
        Validation scores for different training set sizes
    train_sizes : np.ndarray
        Training set sizes
    title : str, default='Learning Curves'
        Plot title
    figsize : tuple, default=(10, 6)
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    ax.plot(train_sizes, train_mean, "o-", color="blue", label="Training Score")
    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
        color="blue",
    )

    ax.plot(train_sizes, val_mean, "o-", color="red", label="Validation Score")
    ax.fill_between(
        train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="red"
    )

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


def plot_roc_curve(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    title: str = "ROC Curve",
    figsize: tuple = (8, 8),
) -> plt.Figure:
    """
    Plot ROC curve for binary classification models.

    Parameters
    ----------
    y_true : pd.Series
        True binary class labels
    y_pred_proba : np.ndarray
        Predicted probabilities for the positive class (shape: [n_samples])
    title : str, default='ROC Curve'
        Plot title
    figsize : tuple, default=(8, 8)
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot ROC curve
    ax.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"ROC curve (AUC = {roc_auc:.4f})",
    )

    # Plot diagonal line (random classifier)
    ax.plot(
        [0, 1],
        [0, 1],
        color="navy",
        lw=2,
        linestyle="--",
        label="Random (AUC = 0.5000)",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig
