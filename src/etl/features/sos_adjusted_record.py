"""SOS-Adjusted Record feature calculation.

Adjusts a team's win percentage by the difficulty of its schedule using
multiplicative scaling:

    sos_adj = raw_win_pct * (team_sos / league_avg_sos) ^ alpha

Teams facing tougher opponents (SOS above league average) get their win
percentage scaled up; those facing weaker opponents get scaled down.
"""

import numpy as np
import pandas as pd


def calculate_sos_adjusted_record(
    records_df: pd.DataFrame,
    sos_df: pd.DataFrame,
    record_lags: list[int],
    sos_lags: list[int],
    alpha: float = 1.0,
) -> pd.DataFrame:
    """
    Calculate SOS-adjusted win percentages.

    Parameters
    ----------
    records_df : pd.DataFrame
        Output of ``calculate_record()``.  Must contain columns:
        gameId, gameDate, season, teamId, total_wins, games_played,
        and ``record_L{lag}`` for each lag in *record_lags*.
    sos_df : pd.DataFrame
        Output of ``calculate_strength_of_schedule()``.  Must contain
        columns: gameId, season, teamId, gameDate, and ``sos_L{lag}``
        for each lag in *sos_lags*.
    record_lags : list[int]
        Lag windows present in *records_df*.
    sos_lags : list[int]
        Lag windows present in *sos_df*.
    alpha : float, default 1.0
        Scaling exponent.  Higher values amplify the SOS adjustment.

    Returns
    -------
    pd.DataFrame
        Team-level DataFrame with columns:
        gameId, season, teamId, gameDate,
        and ``sos_adj_record_L{lag}`` for each lag in the intersection
        of *record_lags* and *sos_lags*.
    """
    join_keys = ["gameId", "season", "teamId", "gameDate"]

    # Select columns needed from records
    record_cols = [f"record_L{lag}" for lag in record_lags if f"record_L{lag}" in records_df.columns]
    records_subset = records_df[join_keys + record_cols].copy()

    # Merge records and SOS on team-game keys
    merged = records_subset.merge(sos_df, on=join_keys, how="inner")

    matched_lags = sorted(set(record_lags) & set(sos_lags))

    result_cols = []

    # --- sos_adj_record_L{lag} for each matched lag ---
    for lag in matched_lags:
        sos_col = f"sos_L{lag}"
        record_col = f"record_L{lag}"
        adj_col = f"sos_adj_record_L{lag}"

        league_avg_lag = merged.groupby(["season", "gameDate"])[sos_col].transform("mean")
        merged[adj_col] = _apply_adjustment(
            merged[record_col], merged[sos_col], league_avg_lag, alpha,
        ).round(4)
        result_cols.append(adj_col)

    return merged[join_keys + result_cols].copy()


def _apply_adjustment(
    raw: pd.Series,
    team_sos: pd.Series,
    league_avg_sos: pd.Series,
    alpha: float,
) -> pd.Series:
    """Apply multiplicative SOS adjustment with safe division."""
    safe_avg = league_avg_sos.replace(0, np.nan)
    return raw * (team_sos / safe_avg) ** alpha
