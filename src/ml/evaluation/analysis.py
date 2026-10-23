"""Metadata-based breakdowns: join predictions with game metadata, no figures.

The figures drawn from these live in :mod:`src.ml.plotting.breakdowns`.
"""

from typing import Optional

import numpy as np
import pandas as pd


def build_analysis_df(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Align predictions with metadata into a single DataFrame."""
    df = metadata.copy()
    df["y_true"] = np.asarray(y_true)
    df["y_pred"] = y_pred
    df["correct"] = (df["y_true"] == df["y_pred"]).astype(int)
    if y_pred_proba is not None:
        df["y_proba"] = y_pred_proba
    return df


def compute_season_phase_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    metadata: pd.DataFrame,
    bins: list[int] | None = None,
) -> pd.DataFrame:
    """Compute accuracy stratified by season phase (games-played buckets).

    Uses ``min(games_played_HT, games_played_VT)`` to assign each observation
    to a phase bucket.

    Parameters
    ----------
    y_true : pd.Series
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    metadata : pd.DataFrame
        Must contain ``games_played_HT`` and ``games_played_VT``.
    bins : list[int], optional
        Bin edges. Defaults to 10-game buckets from 0 to 90.

    Returns
    -------
    pd.DataFrame
        Columns: ``phase``, ``accuracy``, ``n_games``.
    """
    if bins is None:
        bins = list(range(0, 81, 10)) + [82]

    df = build_analysis_df(y_true, y_pred, None, metadata)

    if "games_played_HT" not in df.columns or "games_played_VT" not in df.columns:
        raise ValueError("metadata must contain 'games_played_HT' and 'games_played_VT'")

    min_gp = np.minimum(
        pd.to_numeric(df["games_played_HT"], errors="coerce"),
        pd.to_numeric(df["games_played_VT"], errors="coerce"),
    )

    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
    df["phase"] = pd.cut(min_gp + 1, bins=bins, labels=labels, include_lowest=True)

    phase_stats = (
        df.groupby("phase", observed=True)["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n_games"})
        .reset_index()
    )
    phase_stats.rename(columns={"phase": "season_phase"}, inplace=True)
    return phase_stats
