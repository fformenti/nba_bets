"""Figure functions for model evaluation.

Everything here takes arrays or tables and returns a figure. Nothing here
trains, scores or reads a model: the inputs come from a run's result files
(see :mod:`src.ml.plotting.report`).
"""

from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from typing import Optional
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Colours for series compared against the model. `seagreen` is first because
# Polymarket is the comparison that matters, and it reads as the market in every
# figure that carries one.
COMPARISON_COLORS = ["seagreen", "tomato", "mediumpurple"]


def stamp_timestamp(fig: plt.Figure) -> None:
    """Stamp the figure with its generation time so staleness is visible on the image itself."""
    fig.text(
        0.995, 0.005,
        f"Generated {datetime.now():%Y-%m-%d %H:%M:%S}",
        ha="right", va="bottom", fontsize=7, color="gray", alpha=0.6,
    )


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
    stamp_timestamp(fig)

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
    stamp_timestamp(fig)

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
    stamp_timestamp(fig)

    return fig


def plot_feature_importance(
    importance: pd.DataFrame,
    top_n: int = 20,
    title: str = "Feature Importance",
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Horizontal bars of the ``top_n`` largest importances.

    ``importance`` has columns ``feature`` and ``importance`` - the shape of a
    run's ``model/feature_importance.csv``.
    """
    importance_df = importance.sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(range(len(importance_df)), importance_df["importance"])
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df["feature"])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    stamp_timestamp(fig)

    return fig


def plot_roc_curve(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    title: str = "ROC Curve",
    figsize: tuple = (8, 8),
    model_label: str = "ROC curve",
    reference_curves: Optional[dict] = None,
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
    model_label : str, default='ROC curve'
        Legend label for the model's curve.
    reference_curves : dict, optional
        ``{label: probabilities}`` for curves to compare the model against — the
        market, say. Every series must be scored on the same rows as
        ``y_true``; a curve drawn from a different slate of games is not
        comparable to the model's, whatever the legend claims.

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
        color="steelblue" if reference_curves else "darkorange",
        lw=2,
        label=f"{model_label} (AUC = {roc_auc:.4f})",
    )

    for (label, proba), color in zip(
        (reference_curves or {}).items(), cycle(COMPARISON_COLORS)
    ):
        ref_fpr, ref_tpr, _ = roc_curve(y_true, proba)
        ax.plot(
            ref_fpr,
            ref_tpr,
            color=color,
            lw=2,
            label=f"{label} (AUC = {auc(ref_fpr, ref_tpr):.4f})",
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
    stamp_timestamp(fig)

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
    stamp_timestamp(fig)

    return fig


def _reliability_diagram_points(
    y_true,
    y_pred: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean predicted value, fraction of positives, and counts per probability bin.

    Bins are ``[0, 1/n_bins)``, …, ``[(n_bins-1)/n_bins, 1]`` — left-closed,
    last bin right-closed so 1.0 is included. Empty bins are omitted.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    bin_edges = np.linspace(0, 1, n_bins + 1)
    mean_pred: list[float] = []
    frac_pos: list[float] = []
    counts: list[int] = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_pred >= lo) & (y_pred <= hi)
        else:
            mask = (y_pred >= lo) & (y_pred < hi)
        if not mask.any():
            continue
        mean_pred.append(float(y_pred[mask].mean()))
        frac_pos.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))
    return (
        np.asarray(mean_pred),
        np.asarray(frac_pos),
        np.asarray(counts, dtype=int),
    )


def plot_calibration_curve(
    y_true,
    y_pred_proba_uncalibrated: np.ndarray,
    y_pred_proba_calibrated: Optional[np.ndarray] = None,
    calibration_method: str = "sigmoid",
    n_bins: int = 10,
    title: str = "Calibration Curve (Reliability Diagram)",
    figsize: tuple = (12, 7),
    y_pred_proba_market: Optional[np.ndarray] = None,
    market_label: str = "Polymarket",
    annotate_counts: bool = True,
) -> plt.Figure:
    """Plot reliability diagrams for whichever probability series are given.

    ``y_pred_proba_uncalibrated`` is always drawn. The calibrated series and the
    market series are each optional, so this renders a model-vs-market figure on
    a run where no calibrator was adopted — the market comparison does not depend
    on calibration having happened.

    ``annotate_counts`` labels every point with its bin count. Turn it off once a
    third series is on the axes: the curves track each other closely by
    construction, so three sets of counts land on top of one another and on top
    of the curves. The two-series figure keeps them.

    With default ``n_bins=10``, bins are width 0.1: ``[0, 0.1)``, …, ``[0.9, 1.0]``.
    """
    from src.ml.evaluation.metrics import compute_brier_score

    # (label, probabilities, colour, marker, annotation y-offset)
    series = [
        (
            "Uncalibrated",
            y_pred_proba_uncalibrated,
            "steelblue",
            "o-",
            10,
        )
    ]
    if y_pred_proba_calibrated is not None:
        series.append(
            (
                f"Calibrated [{calibration_method}]",
                y_pred_proba_calibrated,
                "darkorange",
                "s-",
                -12,
            )
        )
    if y_pred_proba_market is not None:
        series.append((market_label, y_pred_proba_market, "seagreen", "^-", -24))

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        [0, 1], [0, 1], "--", color="gray", lw=1.5, label="Perfectly Calibrated"
    )

    for label, proba, color, marker, offset in series:
        mean_pred, fraction_pos, n_per_bin = _reliability_diagram_points(
            y_true, proba, n_bins
        )
        brier = compute_brier_score(y_true, proba)
        ax.plot(
            mean_pred, fraction_pos, marker,
            color=color, lw=2,
            label=f"{label} (Brier={brier:.4f})",
        )
        if annotate_counts:
            for pred_val, frac_val, n in zip(mean_pred, fraction_pos, n_per_bin):
                ax.annotate(
                    f"n={n}", (pred_val, frac_val),
                    textcoords="offset points", xytext=(0, offset),
                    ha="center", fontsize=8, color=color,
                )

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    stamp_timestamp(fig)
    return fig


def plot_reliability_comparison(
    y_true,
    series: dict,
    n_bins: int = 10,
    title: str = "Calibration Comparison",
    figsize: tuple = (10, 8),
    dashed: Optional[set] = None,
) -> plt.Figure:
    """Reliability diagram for any number of ``{label: probabilities}`` series.

    Every series must be scored on the rows of ``y_true``. Marker area follows
    each bin's share of games, since sparse bins are noisy. ``dashed`` names
    series drawn dashed and faded - a raw curve next to its recalibrated twin.
    """
    from src.ml.evaluation.metrics import compute_brier_score

    dashed = dashed or set()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], ":", color="gray", lw=1.5, label="Perfectly Calibrated")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for (label, proba), color in zip(series.items(), cycle(colors)):
        mean_pred, fraction_pos, n_per_bin = _reliability_diagram_points(y_true, proba, n_bins)
        is_dashed = label in dashed
        alpha = 0.45 if is_dashed else 1.0
        brier = compute_brier_score(y_true, proba)
        ax.plot(
            mean_pred, fraction_pos, "--" if is_dashed else "-",
            color=color, lw=2, alpha=alpha, label=f"{label} (Brier={brier:.4f})",
        )
        ax.scatter(
            mean_pred, fraction_pos,
            s=20 + 400 * n_per_bin / max(n_per_bin.sum(), 1), color=color, alpha=alpha,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean Predicted P(home win)", fontsize=12)
    ax.set_ylabel("Actual Home-Win Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    stamp_timestamp(fig)
    return fig


def plot_boruta_shap_results(
    results: dict,
    top_n: int = 30,
    title: str = "Boruta-SHAP Feature Selection",
    figsize: tuple = (12, 10),
) -> plt.Figure:
    """Plot Boruta-SHAP results as a color-coded horizontal bar chart.

    ``results`` is a run's ``model/feature_selection.json``, i.e.
    :meth:`BorutaShapSelector.to_dict`.
    """
    support = {
        **{f: "confirmed" for f in results["confirmed_features"]},
        **{f: "rejected" for f in results["rejected_features"]},
        **{f: "tentative" for f in results["tentative_features"]},
    }
    importances = results["feature_importances"]
    max_shadow = results.get("max_shadow_importances") or []

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
    if max_shadow:
        avg_shadow = np.mean(max_shadow)
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
    stamp_timestamp(fig)
    return fig


def plot_shap_summary(
    shap_frame: pd.DataFrame,
    title: str = "SHAP Feature Contribution",
    top_n: int = 20,
    figsize: tuple = (12, 10),
) -> plt.Figure:
    """SHAP beeswarm from a run's ``model/shap.csv``.

    ``shap_frame`` is long: one row per (game_id, feature) with ``shap_value``
    and the preprocessed ``feature_value``. The values are computed at training
    time by :func:`src.ml.utils.shap.compute_shap_frame`; this only draws them.
    """
    import shap

    shap_values = shap_frame.pivot(index="game_id", columns="feature", values="shap_value")
    feature_values = shap_frame.pivot(index="game_id", columns="feature", values="feature_value")
    feature_values = feature_values[shap_values.columns]

    # Limit to top_n features by mean |SHAP|
    mean_abs = shap_values.abs().mean(axis=0).sort_values(ascending=False)
    top = mean_abs.index[: min(top_n, len(mean_abs))]
    top_n = len(top)

    fig, ax = plt.subplots(figsize=figsize)
    plt.sca(ax)
    shap.summary_plot(
        shap_values[top].to_numpy(dtype=np.float64),
        feature_values[top].to_numpy(dtype=np.float64),
        feature_names=list(top),
        show=False,
        plot_size=None,
        cmap=plt.cm.viridis,
    )

    # Overlay mean |SHAP| annotations at the right edge of the plot
    # SHAP beeswarm places the top feature at the highest y position (top_n - 1)
    x_max = ax.get_xlim()[1]
    for i, val in enumerate(mean_abs[top]):
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
    stamp_timestamp(fig)
    return fig
