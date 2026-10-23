"""Metrics computed from a predictions frame, the same way for every family.

``metrics/<split>.json`` and every comparison table come from
:func:`score_predictions`. The sklearn experiment and the LLM diagnostics each
used to score with their own code, so a gap between the two could come from the
scoring rather than from the models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from src.ml.evaluation.metrics import compute_brier_score, compute_ece
from src.ml.results.schema import SERIES_KEY

# Clip before log-loss so one confidently wrong game is finite. Matches
# src/ml/llm/diagnostics.summarize, which the LLM scoreboard was built on.
PROBABILITY_CLIP = 1e-6


def score_series(frame: pd.DataFrame) -> dict[str, Any]:
    """Metrics for one series. Probability metrics need every row to carry ``p_home``."""
    y_true = frame["y_true"].to_numpy(dtype=int)
    metrics: dict[str, Any] = {
        "n_games": len(frame),
        "accuracy": float((frame["y_pred"].to_numpy(dtype=int) == y_true).mean()),
        "home_pick_rate": float(frame["y_pred"].mean()),
        "home_win_rate": float(y_true.mean()),
    }

    p_home = frame["p_home"]
    if len(frame) and p_home.notna().all():
        p = p_home.to_numpy(dtype=float)
        metrics["brier_score"] = compute_brier_score(y_true, p)
        metrics["log_loss"] = float(
            log_loss(y_true, np.clip(p, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP), labels=[0, 1])
        )
        metrics["ece"] = compute_ece(y_true, p)
        if len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, p))

    margins = frame[["pred_margin", "true_margin"]]
    if len(frame) and margins.notna().all().all():
        errors = margins["pred_margin"] - margins["true_margin"]
        metrics["mae"] = float(errors.abs().mean())
        metrics["rmse"] = float(np.sqrt((errors**2).mean()))

    return metrics


def score_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, calibration) in ``frame``, in the order they appear."""
    rows = [
        {"model": model, "calibration": calibration, **score_series(group)}
        for (model, calibration), group in frame.groupby(SERIES_KEY, sort=False)
    ]
    return pd.DataFrame(rows)
