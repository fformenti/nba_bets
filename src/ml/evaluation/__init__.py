"""Model evaluation: metrics and breakdown tables. Figures live in src.ml.plotting."""

from .analysis import build_analysis_df, compute_season_phase_metrics
from .metrics import (
    compute_brier_score,
    compute_classification_metrics,
    compute_ece,
    compute_regression_metrics,
    format_metrics_line,
    get_model_display_name,
    print_metrics_summary,
)

__all__ = [
    "build_analysis_df",
    "compute_brier_score",
    "compute_classification_metrics",
    "compute_ece",
    "compute_regression_metrics",
    "compute_season_phase_metrics",
    "format_metrics_line",
    "get_model_display_name",
    "print_metrics_summary",
]
