"""Visualization utilities for model evaluation."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from sklearn.base import BaseEstimator
from sklearn.calibration import calibration_curve
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


def plot_prediction_accuracy_by_bin(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    title: str = "Prediction Accuracy by Probability Bin",
    figsize: tuple = (12, 7),
) -> plt.Figure:
    """
    Plot model accuracy grouped by 5% prediction probability bins.

    For each bin, accuracy is the fraction of correct predictions where the
    predicted class is home (proba >= 0.5) or away (proba < 0.5).
    """
    bins = np.arange(0, 1.05, 0.05)
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

    midpoints = []
    accuracies = []
    counts = []

    for i in range(len(bins) - 1):
        mask = bin_indices == i
        n = mask.sum()
        if n == 0:
            continue

        bin_true = np.array(y_true)[mask]
        bin_proba = y_pred_proba[mask]
        bin_pred = (bin_proba >= 0.5).astype(int)
        acc = (bin_pred == bin_true).mean()

        midpoint = (bins[i] + bins[i + 1]) / 2
        midpoints.append(midpoint)
        accuracies.append(acc)
        counts.append(n)

    midpoints = np.array(midpoints)
    accuracies = np.array(accuracies)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(midpoints, accuracies, "o-", color="steelblue", lw=2, label="Model Accuracy")

    # Reference line: expected accuracy for a perfectly calibrated model
    ref_x = np.linspace(0, 1, 100)
    ref_y = np.maximum(ref_x, 1 - ref_x)
    ax.plot(ref_x, ref_y, "--", color="gray", lw=1.5, label="Perfect Calibration")

    # Annotate sample counts
    for x, y, n in zip(midpoints, accuracies, counts):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, color="dimgray")

    ax.set_xlabel("Predicted Probability (Home Win)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim([0, 1])
    ax.set_ylim([0.4, 1.05])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


def plot_calibration_curve(
    y_true,
    y_pred_proba_uncalibrated: np.ndarray,
    y_pred_proba_calibrated: np.ndarray,
    calibration_method: str = "sigmoid",
    n_bins: int = 10,
    title: str = "Calibration Curve (Reliability Diagram)",
    figsize: tuple = (12, 7),
) -> plt.Figure:
    """Plot calibration curves comparing uncalibrated vs calibrated probabilities."""
    from src.ml.evaluation.metrics import compute_brier_score

    fraction_pos_uncal, mean_pred_uncal = calibration_curve(
        y_true, y_pred_proba_uncalibrated, n_bins=n_bins, strategy="uniform"
    )
    fraction_pos_cal, mean_pred_cal = calibration_curve(
        y_true, y_pred_proba_calibrated, n_bins=n_bins, strategy="uniform"
    )

    brier_uncal = compute_brier_score(y_true, y_pred_proba_uncalibrated)
    brier_cal = compute_brier_score(y_true, y_pred_proba_calibrated)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        [0, 1], [0, 1], "--", color="gray", lw=1.5, label="Perfectly Calibrated"
    )
    ax.plot(
        mean_pred_uncal, fraction_pos_uncal, "o-",
        color="steelblue", lw=2,
        label=f"Uncalibrated (Brier={brier_uncal:.4f})",
    )
    ax.plot(
        mean_pred_cal, fraction_pos_cal, "s-",
        color="darkorange", lw=2,
        label=f"Calibrated [{calibration_method}] (Brier={brier_cal:.4f})",
    )

    # Annotate bin counts for uncalibrated
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for pred_val, frac_val in zip(mean_pred_uncal, fraction_pos_uncal):
        bin_idx = np.clip(np.digitize(pred_val, bin_edges) - 1, 0, n_bins - 1)
        mask = (y_pred_proba_uncalibrated >= bin_edges[bin_idx]) & (
            y_pred_proba_uncalibrated < bin_edges[bin_idx + 1]
        )
        n = mask.sum()
        ax.annotate(
            f"n={n}", (pred_val, frac_val),
            textcoords="offset points", xytext=(0, 10),
            ha="center", fontsize=8, color="dimgray",
        )

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_boruta_shap_results(
    selector,
    top_n: int = 30,
    title: str = "Boruta-SHAP Feature Selection",
    figsize: tuple = (12, 10),
) -> plt.Figure:
    """Plot Boruta-SHAP results as a color-coded horizontal bar chart.

    Parameters
    ----------
    selector : BorutaShapSelector
        Fitted selector with feature_importances_ and get_support().
    top_n : int
        Number of top features to display.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    support = selector.get_support()
    importances = selector.feature_importances_

    # Sort by importance descending, take top_n
    sorted_features = sorted(importances, key=importances.get, reverse=True)[:top_n]

    # Reverse for horizontal bar chart (top feature at top)
    sorted_features = sorted_features[::-1]
    values = [importances[f] for f in sorted_features]

    color_map = {"confirmed": "#2ecc71", "rejected": "#e74c3c", "tentative": "#f1c40f"}
    colors = [color_map.get(support.get(f, "tentative"), "#95a5a6") for f in sorted_features]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(sorted_features)), values, color=colors)
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel("Mean |SHAP| Importance")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Max shadow threshold line (average across iterations)
    if selector.max_shadow_importances_:
        avg_shadow = np.mean(selector.max_shadow_importances_)
        ax.axvline(x=avg_shadow, color="gray", linestyle="--", lw=1.5, label=f"Avg max shadow ({avg_shadow:.4f})")
        ax.legend(fontsize=10)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Confirmed"),
        Patch(facecolor="#e74c3c", label="Rejected"),
        Patch(facecolor="#f1c40f", label="Tentative"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.tight_layout()
    return fig


def plot_shap_summary(
    model,
    X_transformed: np.ndarray,
    feature_names: list,
    title: str = "SHAP Feature Contribution",
    top_n: int = 20,
    max_samples: int = 500,
    figsize: tuple = (12, 10),
) -> plt.Figure:
    """
    Generate a SHAP beeswarm summary plot for tree-based models.

    Parameters
    ----------
    model : tree-based estimator (RF, GB, XGB, LGBM)
    X_transformed : np.ndarray
        Preprocessed feature matrix (already passed through the preprocessor)
    feature_names : list
        Feature names corresponding to X_transformed columns
    title : str
        Plot title
    top_n : int
        Number of top features to display
    max_samples : int
        Max rows to use for SHAP computation (subsampled for performance)
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    import shap

    if hasattr(X_transformed, "values"):
        X_transformed = X_transformed.values
    X_transformed = np.asarray(X_transformed, dtype=np.float64)

    # Subsample for performance
    if X_transformed.shape[0] > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(X_transformed.shape[0], max_samples, replace=False)
        X_transformed = X_transformed[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    # For binary classification some models return list [neg_class, pos_class]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    # Newer SHAP returns 3D array (n_samples, n_features, n_classes) for tree ensembles
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    # Clamp top_n to actual number of features
    n_features = shap_values.shape[1]
    top_n = min(top_n, n_features)

    # Limit to top_n features by mean |SHAP|
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    shap_values_top = shap_values[:, top_idx]
    X_top = X_transformed[:, top_idx]
    feature_names_top = [feature_names[i] for i in top_idx]

    # Mean |SHAP| per top feature (sorted descending = top feature at y=0)
    mean_abs_top = mean_abs[top_idx]

    fig, ax = plt.subplots(figsize=figsize)
    plt.sca(ax)
    shap.summary_plot(
        shap_values_top,
        X_top,
        feature_names=feature_names_top,
        show=False,
        plot_size=None,
        cmap=plt.cm.viridis,
    )

    # Overlay mean |SHAP| annotations at the right edge of the plot
    # SHAP beeswarm places the top feature at the highest y position (top_n - 1)
    x_max = ax.get_xlim()[1]
    for i, val in enumerate(mean_abs_top):
        y_pos = top_n - 1 - i
        ax.annotate(
            f"avg |SHAP|={val:.4f}",
            (x_max, y_pos),
            fontsize=7.5,
            va="center",
            ha="right",
            color="black",
            fontweight="bold",
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig
