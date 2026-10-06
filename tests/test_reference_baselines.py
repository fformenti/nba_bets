"""The two reference rows the model gets read against: the floor and the market.

`home_win_baseline` is the floor — beating the record baseline means little if
home advantage alone already gets there. `polymarket_baseline` is the ceiling.
Both are scored through the same `ModelTrainer` path as every trained family, so
these tests pin the estimator contract that path depends on.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml.evaluation.metrics import compute_home_win_metrics, format_metrics_line
from src.ml.models.baseline import HomeWinBaseline, MarketProbabilityBaseline


def test_home_win_baseline_always_picks_the_home_team():
    X = pd.DataFrame({"anything": range(10)})
    y = pd.Series([1] * 6 + [0] * 4)

    model = HomeWinBaseline(home_win_rate=0.6).fit(X, y)

    assert model.predict(X).tolist() == [1] * 10
    assert model.score(X, y) == pytest.approx(0.6)


def test_home_win_baseline_emits_the_rate_not_a_hard_one():
    """A constant 1.0 would make its Brier incomparable to the other rows."""
    X = pd.DataFrame({"anything": range(4)})
    proba = HomeWinBaseline(home_win_rate=0.58).fit(X).predict_proba(X)

    assert proba[:, 1].tolist() == [0.58] * 4
    assert proba[:, 0].tolist() == pytest.approx([0.42] * 4)


def test_home_win_baseline_roc_auc_is_one_half():
    """A constant score cannot discriminate; 0.5 is the honest reading."""
    from sklearn.metrics import roc_auc_score

    X = pd.DataFrame({"anything": range(8)})
    y = pd.Series([1, 0, 1, 1, 0, 1, 0, 1])
    proba = HomeWinBaseline(home_win_rate=0.58).fit(X, y).predict_proba(X)[:, 1]

    assert roc_auc_score(y, proba) == pytest.approx(0.5)


def _market(prices: dict) -> pd.Series:
    return pd.Series(prices, index=pd.Index(prices.keys(), dtype="Int64"))


def test_market_baseline_calls_its_favourite():
    market = _market({1: 0.905, 2: 0.465, 3: 0.5})
    X = pd.DataFrame({"gameId": pd.array([1, 2, 3], dtype="Int64")})

    model = MarketProbabilityBaseline(market).fit(X)

    # 0.5 is a home call: the tie has to break somewhere and >= matches the
    # market_prediction rule in polymarket_baseline.compare_to_market.
    assert model.predict(X).tolist() == [1, 0, 1]
    assert model.predict_proba(X)[:, 1].tolist() == pytest.approx([0.905, 0.465, 0.5])


def test_market_baseline_refuses_an_unpriced_game():
    """Imputing 0.5 would enter a coin flip the market never actually held."""
    market = _market({1: 0.7})
    X = pd.DataFrame({"gameId": pd.array([1, 99], dtype="Int64")})

    with pytest.raises(ValueError, match="no market price"):
        MarketProbabilityBaseline(market).fit(X)


def test_home_win_metrics_scores_roc_auc_when_probabilities_are_present():
    frame = pd.DataFrame(
        {
            "actual_home_win": [1, 0, 1, 0],
            "prediction": [1, 0, 1, 1],
            "home_win_probability": [0.9, 0.2, 0.8, 0.6],
        }
    )

    metrics = compute_home_win_metrics(frame)

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["n_games"] == 4


def test_home_win_metrics_omits_roc_auc_for_a_hard_call_only_source():
    frame = pd.DataFrame(
        {"actual_home_win": [1, 0, 1, 0], "prediction": [1, 0, 1, 1]}
    )

    metrics = compute_home_win_metrics(frame)

    assert "roc_auc" not in metrics
    assert "brier_score" not in metrics


def test_metrics_line_flags_a_row_scored_on_a_subset():
    line = format_metrics_line(
        {"test_accuracy": 0.62, "test_roc_auc": 0.68, "test_n_games": 1180},
        prefix="test",
    )

    assert line.endswith("n=1180")


def test_metrics_line_omits_the_count_when_absent():
    line = format_metrics_line({"test_accuracy": 0.62}, prefix="test")

    assert "n=" not in line
