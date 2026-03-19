"""Model evaluation and visualization utilities."""

from .analysis import (
    generate_analysis,
    plot_accuracy_by_season,
    plot_conference_matchup_accuracy,
    plot_error_by_team,
    plot_neutral_court_accuracy,
    plot_overtime_accuracy,
    plot_score_margin_calibration,
)
from .metrics import (
    compute_brier_score,
    compute_classification_metrics,
    compute_ece,
    compute_regression_metrics,
    format_metrics_line,
    get_conference_display_name,
    get_model_display_name,
    print_metrics_summary,
)
from .visualization import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_learning_curves,
    plot_prediction_accuracy_by_bin,
    plot_predictions,
    plot_residuals,
    plot_roc_curve,
    plot_shap_summary,
)

__all__ = [
    "compute_brier_score",
    "compute_classification_metrics",
    "compute_ece",
    "compute_regression_metrics",
    "format_metrics_line",
    "generate_analysis",
    "get_conference_display_name",
    "get_model_display_name",
    "plot_accuracy_by_season",
    "plot_calibration_curve",
    "plot_conference_matchup_accuracy",
    "plot_confusion_matrix",
    "plot_error_by_team",
    "plot_feature_importance",
    "plot_learning_curves",
    "plot_neutral_court_accuracy",
    "plot_overtime_accuracy",
    "plot_prediction_accuracy_by_bin",
    "plot_predictions",
    "plot_residuals",
    "plot_roc_curve",
    "plot_shap_summary",
    "plot_score_margin_calibration",
    "print_metrics_summary",
]
