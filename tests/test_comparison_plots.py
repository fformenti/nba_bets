"""The evaluation figures that put the model next to something else.

These plots are the only place the model-vs-market question is *read* rather
than logged, so the tests are about what ends up on the axes: one line per
series, a table that says the same thing as the figure, and — the regression
that would otherwise be silent — a calibration figure that still renders on a
run where no calibrator was adopted.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from src.ml.plotting.breakdowns import plot_season_phase_accuracy
from src.ml.plotting.charts import plot_calibration_curve, plot_roc_curve


@pytest.fixture
def outcomes():
    """A test-split-shaped frame: alternating winners over two season phases."""
    rng = np.random.RandomState(0)
    n = 200
    y_true = pd.Series(rng.randint(0, 2, n))
    metadata = pd.DataFrame(
        {
            "games_played_HT": np.repeat([15, 45], n // 2),
            "games_played_VT": np.repeat([18, 48], n // 2),
        }
    )
    return y_true, metadata


def _labelled_lines(fig):
    return [line.get_label() for line in fig.axes[0].get_lines()]


# ---------------------------------------------------------------------------
# ROC
# ---------------------------------------------------------------------------


def test_roc_draws_one_curve_per_reference_series(outcomes):
    y_true, _ = outcomes
    proba = np.linspace(0.05, 0.95, len(y_true))

    fig = plot_roc_curve(
        y_true, proba, reference_curves={"Polymarket": proba[::-1]}, model_label="xgboost"
    )

    labels = _labelled_lines(fig)
    # Model, market, and the random diagonal.
    assert len(labels) == 3
    assert any(label.startswith("xgboost (AUC =") for label in labels)
    assert any(label.startswith("Polymarket (AUC =") for label in labels)


def test_roc_without_references_is_unchanged(outcomes):
    """The full-test figure must not move when the market twin is added."""
    y_true, _ = outcomes
    proba = np.linspace(0.05, 0.95, len(y_true))

    fig = plot_roc_curve(y_true, proba)

    labels = _labelled_lines(fig)
    assert len(labels) == 2
    assert labels[0].startswith("ROC curve (AUC =")
    assert fig.axes[0].get_lines()[0].get_color() == "darkorange"


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_calibration_renders_against_the_market_without_a_calibrator(outcomes):
    """No calibrator adopted is not a reason to lose the market comparison."""
    y_true, _ = outcomes
    proba = np.linspace(0.05, 0.95, len(y_true))

    fig = plot_calibration_curve(
        y_true, proba, None, y_pred_proba_market=proba[::-1]
    )

    labels = _labelled_lines(fig)
    assert any(label.startswith("Uncalibrated (Brier=") for label in labels)
    assert any(label.startswith("Polymarket (Brier=") for label in labels)
    assert not any(label.startswith("Calibrated") for label in labels)


def test_calibration_draws_all_three_when_a_calibrator_ran(outcomes):
    y_true, _ = outcomes
    proba = np.linspace(0.05, 0.95, len(y_true))

    fig = plot_calibration_curve(
        y_true,
        proba,
        np.clip(proba + 0.02, 0, 1),
        calibration_method="isotonic",
        y_pred_proba_market=proba[::-1],
    )

    labels = _labelled_lines(fig)
    # Diagonal + three series.
    assert len(labels) == 4
    assert any(label.startswith("Calibrated [isotonic] (Brier=") for label in labels)


def test_calibration_keeps_working_on_the_old_positional_call(outcomes):
    """`experiment.py` calls this with three positional arguments."""
    y_true, _ = outcomes
    proba = np.linspace(0.05, 0.95, len(y_true))

    fig = plot_calibration_curve(y_true, proba, np.clip(proba + 0.02, 0, 1), "sigmoid")

    labels = _labelled_lines(fig)
    assert len(labels) == 3
    assert not any(label.startswith("Polymarket") for label in labels)


# ---------------------------------------------------------------------------
# Season phase
# ---------------------------------------------------------------------------


def test_season_phase_table_carries_one_column_per_series(outcomes):
    y_true, metadata = outcomes
    y_pred = np.ones(len(y_true), dtype=int)

    _, table = plot_season_phase_accuracy(
        y_true,
        y_pred,
        metadata,
        comparison_preds={
            "Polymarket": np.zeros(len(y_true), dtype=int),
            "Record baseline": y_pred,
        },
    )

    assert list(table.columns) == [
        "season_phase",
        "accuracy_model",
        "n_games",
        "accuracy_polymarket",
        "accuracy_record_baseline",
    ]
    # Always-home and always-away are complements within every bucket.
    complements = table["accuracy_model"] + table["accuracy_polymarket"]
    assert np.allclose(complements, 1.0)


def test_season_phase_without_comparisons_reports_the_model_alone(outcomes):
    y_true, metadata = outcomes

    fig, table = plot_season_phase_accuracy(
        y_true, np.ones(len(y_true), dtype=int), metadata
    )

    assert list(table.columns) == ["season_phase", "accuracy_model", "n_games"]
    # One accuracy line plus its average hline, and nothing else.
    assert _labelled_lines(fig) == ["Model accuracy", "Model avg (50.5%)"]
