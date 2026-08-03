"""Point differential feature calculations."""

import pandas as pd


def calculate_home_pts_diff(games, lags=[]):
    """Calculate point differential for home teams."""
    home_games = games.rename(
        index=str,
        columns={
            "hometeamId": "teamId",
        },
    )
    return calculate_pts_diff(home_games, lags)


def calculate_away_pts_diff(games, lags=[]):
    """Calculate point differential for away teams."""
    # Work on a copy to avoid modifying the original DataFrame
    games_aux = games.copy()
    games_aux["pts_diff"] = games_aux["pts_diff"] * -1

    away_games = games_aux.rename(
        index=str,
        columns={
            "awayteamId": "teamId",
        },
    )

    return calculate_pts_diff(away_games, lags)


def calculate_pts_diff(games, lags=[]):
    """
    Calculate point differential statistics with rolling windows.

    Parameters
    ----------
    games : pd.DataFrame
        Games DataFrame with teamId, gameDate, season, pts_diff columns

    Returns
    -------
    pd.DataFrame
        DataFrame with point differential statistics
    """
    # Sort keys must lead with the groupby keys, and gameId breaks ties so games
    # sharing a tip-off timestamp order deterministically.
    games = games.sort_values(["teamId", "season", "gameDate", "gameId"])

    games["pts_diff_L1"] = games.groupby(["teamId", "season"])["pts_diff"].shift(1)

    avg_pts_diff_lags_cols = []
    for lag in lags:
        games[f"pts_diff_avg_L{lag}"] = (
            games.groupby(["teamId", "season"])["pts_diff_L1"]
            .rolling(window=lag, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        ).fillna(0.0)
        avg_pts_diff_lags_cols.append(f"pts_diff_avg_L{lag}")

    return_cols = [
        "gameId",
        "gameDate",
        "season",
        "teamId",
        "pts_diff",
    ] + avg_pts_diff_lags_cols
    return games[return_cols].copy()


# ── Normalized point differential ────────────────────────────────────────────


def _add_season_rolling_avg_total_pts(games: pd.DataFrame) -> pd.DataFrame:
    """
    Add a rolling season-to-date average of total points per game (both teams
    combined) to a games DataFrame that already has a unique row per game.

    The average is shifted by 1 so the current game's score is excluded
    (no lookahead leakage). The result column is ``season_avg_total_pts``.
    """
    games = games.copy()
    # total points scored in the game (both teams)
    games["_total_pts"] = games["homeScore"] + games["awayScore"]
    games = games.sort_values(["season", "gameDate"])

    # shift(1) within season so current game is excluded
    games["_total_pts_L1"] = games.groupby("season")["_total_pts"].shift(1)

    # expanding mean = season-to-date average using only past games
    season_to_date = (
        games.groupby("season")["_total_pts_L1"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    # The first game of a season has no season-to-date average yet. Fall back to
    # the PREVIOUS season's average rather than the all-seasons mean: this
    # feature exists to neutralise era scoring pace, so a denominator mixing the
    # 1940s and the 2020s defeats its purpose (and peeks at the future).
    # Seasons sort lexically in chronological order ("1999/00" < "2000/01").
    season_mean = games.groupby("season")["_total_pts"].mean()
    fallback = games["season"].map(season_mean.shift(1))

    # The earliest season in the dataset has no predecessor; use its own mean.
    # This affects only that season's very first game.
    fallback = fallback.fillna(games["season"].map(season_mean))

    games["season_avg_total_pts"] = season_to_date.fillna(fallback)

    games = games.drop(columns=["_total_pts", "_total_pts_L1"])
    return games


def _compute_norm_pts_diff(games: pd.DataFrame, lags: list) -> pd.DataFrame:
    """
    Core computation of normalized point differential rolling averages.

    ``pts_diff`` is divided by the season-to-date average total points per
    game before rolling averages are computed.  This makes the feature
    season-neutral across eras with different scoring pace.
    """
    games = games.sort_values(["teamId", "season", "gameDate", "gameId"])

    games["norm_pts_diff"] = (
        games["pts_diff"] / games["season_avg_total_pts"].replace(0, float("nan"))
    ).fillna(0.0)

    games["norm_pts_diff_L1"] = games.groupby(["teamId", "season"])["norm_pts_diff"].shift(1)

    norm_cols = []
    for lag in lags:
        col = f"norm_pts_diff_avg_L{lag}"
        games[col] = (
            games.groupby(["teamId", "season"])["norm_pts_diff_L1"]
            .rolling(window=lag, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        ).fillna(0.0)
        norm_cols.append(col)

    return_cols = ["gameId", "gameDate", "season", "teamId", "pts_diff"] + norm_cols
    return games[return_cols].copy()


def calculate_norm_home_pts_diff(games: pd.DataFrame, lags: list = []) -> pd.DataFrame:
    """Normalized point differential for home teams (home-games only)."""
    home_games = games.rename(columns={"hometeamId": "teamId"})
    return _compute_norm_pts_diff(home_games, lags)


def calculate_norm_away_pts_diff(games: pd.DataFrame, lags: list = []) -> pd.DataFrame:
    """Normalized point differential for away teams (away-games only)."""
    games_aux = games.copy()
    games_aux["pts_diff"] = games_aux["pts_diff"] * -1
    away_games = games_aux.rename(columns={"awayteamId": "teamId"})
    return _compute_norm_pts_diff(away_games, lags)


def calculate_norm_pts_diff(games: pd.DataFrame, lags: list = []) -> pd.DataFrame:
    """Normalized point differential for all games (home + away combined)."""
    return _compute_norm_pts_diff(games, lags)
