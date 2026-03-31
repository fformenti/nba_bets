"""Game Difficulty Score (GDS) feature calculation.

Assigns a per-game quality score to each team based on the game outcome,
opponent strength, and home/away context:

    Win:  GDS = +Q(opponent) * (1 + β * is_away)
    Loss: GDS = -(1 - Q(opponent)) * (1 + β * is_home)

Where Q is the opponent's SOS-adjusted record (or cumulative win% as fallback).
Rolling averages of GDS at various lag windows become the ML features.
"""

import numpy as np
import pandas as pd


def calculate_game_difficulty_score(
    games: pd.DataFrame,
    sos_adj_record_df: pd.DataFrame,
    lags: list[int],
    beta: float = 0.10,
    min_games: int = 1,
) -> pd.DataFrame:
    """
    Calculate Game Difficulty Score and rolling averages for each team per game.

    Parameters
    ----------
    games : pd.DataFrame
        Game-level DataFrame with columns:
        gameId, gameDate, season, hometeamId, awayteamId, winner.
    sos_adj_record_df : pd.DataFrame
        Output of ``calculate_sos_adjusted_record()``.  Used as opponent
        quality Q(o,g).  Must contain: gameId, season, teamId, gameDate,
        and at least one ``sos_adj_record_L{lag}`` column.  The largest
        available lag is used.  Falls back to cumulative win% when NaN
        or when this DataFrame is empty.
    lags : list[int]
        Rolling window sizes.  Each produces a column ``gds_L{lag}``.
    beta : float, default 0.10
        Home/away adjustment factor.
    min_games : int, default 1
        Minimum games in rolling window to produce a value.

    Returns
    -------
    pd.DataFrame
        Team-level DataFrame with columns:
        gameId, season, teamId, gameDate, gds_L{lag1}, ...
    """
    team_games = _build_team_game_table(games)
    team_games = _add_opponent_quality(team_games, sos_adj_record_df)
    team_games["gds_raw"] = _compute_raw_gds(team_games, beta)

    team_games = _add_rolling_gds(team_games, lags, min_games)

    return_cols = ["gameId", "season", "teamId", "gameDate"] + [
        f"gds_L{lag}" for lag in lags
    ]
    return team_games[return_cols].copy()


def calculate_home_gds(
    games: pd.DataFrame,
    sos_adj_record_df: pd.DataFrame,
    location_lags: list[int],
    beta: float = 0.10,
    min_games: int = 1,
) -> pd.DataFrame:
    """Calculate rolling GDS over home games only."""
    team_games = _build_team_game_table(games)
    team_games = _add_opponent_quality(team_games, sos_adj_record_df)
    team_games["gds_raw"] = _compute_raw_gds(team_games, beta)

    home_only = team_games[team_games["is_home"] == 1].copy()
    home_only = _add_rolling_gds(home_only, location_lags, min_games)

    return_cols = ["gameId", "season", "teamId", "gameDate"] + [
        f"gds_L{lag}" for lag in location_lags
    ]
    return home_only[return_cols].copy()


def calculate_away_gds(
    games: pd.DataFrame,
    sos_adj_record_df: pd.DataFrame,
    location_lags: list[int],
    beta: float = 0.10,
    min_games: int = 1,
) -> pd.DataFrame:
    """Calculate rolling GDS over away games only."""
    team_games = _build_team_game_table(games)
    team_games = _add_opponent_quality(team_games, sos_adj_record_df)
    team_games["gds_raw"] = _compute_raw_gds(team_games, beta)

    away_only = team_games[team_games["is_away"] == 1].copy()
    away_only = _add_rolling_gds(away_only, location_lags, min_games)

    return_cols = ["gameId", "season", "teamId", "gameDate"] + [
        f"gds_L{lag}" for lag in location_lags
    ]
    return away_only[return_cols].copy()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_team_game_table(games: pd.DataFrame) -> pd.DataFrame:
    """Split game-level rows into team-level rows with location flags."""
    base_cols = ["gameId", "gameDate", "season"]

    home = games[base_cols + ["hometeamId", "awayteamId", "winner"]].copy()
    home["teamId"] = home["hometeamId"]
    home["opponentId"] = home["awayteamId"]
    home["win_bool"] = (home["hometeamId"] == home["winner"]).astype(int)
    home["is_home"] = 1
    home["is_away"] = 0

    away = games[base_cols + ["hometeamId", "awayteamId", "winner"]].copy()
    away["teamId"] = away["awayteamId"]
    away["opponentId"] = away["hometeamId"]
    away["win_bool"] = (away["awayteamId"] == away["winner"]).astype(int)
    away["is_home"] = 0
    away["is_away"] = 1

    keep = base_cols + ["teamId", "opponentId", "win_bool", "is_home", "is_away"]
    return (
        pd.concat([home[keep], away[keep]], ignore_index=True)
        .sort_values(["teamId", "season", "gameDate"])
        .reset_index(drop=True)
    )


def _add_opponent_quality(
    team_games: pd.DataFrame,
    sos_adj_record_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add opponent quality Q column, with fallback to cumulative win%."""
    team_games = team_games.copy()

    # Try to get sos_adj_record as primary quality measure
    sos_col = _find_largest_sos_adj_col(sos_adj_record_df)

    if sos_col is not None:
        opp_quality = sos_adj_record_df[
            ["gameId", "season", "teamId", sos_col]
        ].rename(columns={"teamId": "opponentId", sos_col: "opp_sos_adj"})
        team_games = team_games.merge(
            opp_quality,
            on=["gameId", "season", "opponentId"],
            how="left",
        )
    else:
        team_games["opp_sos_adj"] = np.nan

    # Fallback: opponent cumulative win% (shifted to avoid leakage)
    fallback = _build_cum_win_pct_lookup(team_games)
    team_games = team_games.merge(
        fallback.rename(columns={"teamId": "opponentId", "cum_win_pct": "opp_cum_pct"}),
        on=["gameId", "season", "opponentId"],
        how="left",
    )

    # Use sos_adj_record where available, fall back to cum_win_pct
    team_games["opp_quality"] = team_games["opp_sos_adj"].fillna(
        team_games["opp_cum_pct"]
    )
    # Clamp to [0, 1] (sos_adj can slightly exceed 1)
    team_games["opp_quality"] = team_games["opp_quality"].clip(0, 1)

    team_games.drop(columns=["opp_sos_adj", "opp_cum_pct"], inplace=True)
    return team_games


def _find_largest_sos_adj_col(df: pd.DataFrame) -> str | None:
    """Find the sos_adj_record column with the largest lag."""
    if df.empty:
        return None
    sos_cols = [c for c in df.columns if c.startswith("sos_adj_record_L")]
    if not sos_cols:
        return None
    # Extract lag numbers and pick the largest
    lags = {c: int(c.split("_L")[1]) for c in sos_cols}
    return max(lags, key=lags.get)


def _build_cum_win_pct_lookup(team_games: pd.DataFrame) -> pd.DataFrame:
    """Build cumulative win% for each team-game, shifted by 1 (prior games only)."""
    df = team_games[["gameId", "season", "teamId", "gameDate", "win_bool"]].copy()
    df = df.sort_values(["teamId", "season", "gameDate"])

    # Shift win_bool so we only count prior games
    df["win_shifted"] = df.groupby(["teamId", "season"])["win_bool"].shift(1)
    df["cum_wins"] = df.groupby(["teamId", "season"])["win_shifted"].cumsum()
    df["game_num"] = df.groupby(["teamId", "season"])["win_shifted"].cumcount()

    # game_num here is 0-indexed count of non-NaN shifted values processed
    # We need the count of prior games = cumcount of the shifted series
    df["prior_games"] = df.groupby(["teamId", "season"]).cumcount()
    df["cum_win_pct"] = np.where(
        df["prior_games"] > 0,
        df["cum_wins"] / df["prior_games"],
        np.nan,
    )

    return df[["gameId", "season", "teamId", "cum_win_pct"]].copy()


def _compute_raw_gds(team_games: pd.DataFrame, beta: float) -> pd.Series:
    """Apply the GDS formula per game."""
    q = team_games["opp_quality"]
    w = team_games["win_bool"]
    is_away = team_games["is_away"]
    is_home = team_games["is_home"]

    win_score = q * (1 + beta * is_away)
    loss_score = -(1 - q) * (1 + beta * is_home)

    return np.where(w == 1, win_score, loss_score).round(4)


def _add_rolling_gds(
    team_games: pd.DataFrame,
    lags: list[int],
    min_games: int,
) -> pd.DataFrame:
    """Add rolling mean GDS columns, shifted by 1 to prevent leakage."""
    team_games = team_games.sort_values(["teamId", "season", "gameDate"])

    # Shift raw GDS so rolling windows use only prior games
    team_games["gds_shifted"] = team_games.groupby(["teamId", "season"])[
        "gds_raw"
    ].shift(1)

    for lag in lags:
        team_games[f"gds_L{lag}"] = (
            team_games.groupby(["teamId", "season"])["gds_shifted"]
            .rolling(window=lag, min_periods=min_games)
            .mean()
            .round(4)
            .reset_index(level=[0, 1], drop=True)
        )

    team_games.drop(columns=["gds_shifted"], inplace=True)
    return team_games
