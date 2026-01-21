"""Point differential feature calculations."""


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
    games = games.sort_values(["teamId", "gameDate", "season"])

    games["pts_diff_L1"] = games.groupby(["teamId", "season"])["pts_diff"].shift(1)
    # games["pts_diff_avg"] = (
    #     games.groupby(["teamId", "season"])["pts_diff_L1"]
    #     .expanding()
    #     .mean()
    #     .reset_index(level=[0, 1], drop=True)
    # ).fillna(0.0)

    avg_pts_diff_lags_cols = []
    for lag in lags:
        games[f"pts_diff_avg_L{lag}"] = (
            games.groupby(["teamId", "season"])["pts_diff_L1"]
            .rolling(window=lag, min_periods=1)
            .mean()
            .round(2)
            .reset_index(level=[0, 1], drop=True)
        ).fillna(0.0)
        avg_pts_diff_lags_cols.append(f"pts_diff_avg_L{lag}")

    # games["pts_diff_avg_L5"] = (
    #     games.groupby(["teamId", "season"])["pts_diff_L1"]
    #     .rolling(window=5, min_periods=1)
    #     .mean()
    #     .reset_index(level=[0, 1], drop=True)
    # ).fillna(0.0)

    return_cols = [
        "gameId",
        "gameDate",
        "season",
        "teamId",
        "pts_diff",
    ] + avg_pts_diff_lags_cols
    return games[return_cols].copy()
