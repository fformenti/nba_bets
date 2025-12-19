"""Point differential feature calculations."""


def calculate_home_pts_diff(games):
    """Calculate point differential for home teams."""
    home_games = games.rename(
        index=str,
        columns={
            "hometeamId": "teamId",
        },
    )
    return calculate_pts_diff(home_games)


def calculate_away_pts_diff(games):
    """Calculate point differential for away teams."""
    games["pts_diff"] = games["pts_diff"].apply(lambda x: -1 * x)

    home_games = games.rename(
        index=str,
        columns={
            "awayteamId": "teamId",
        },
    )

    return calculate_pts_diff(home_games)


def calculate_pts_diff(games):
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
    games["pts_diff_avg"] = (
        games.groupby(["teamId", "season"])["pts_diff_L1"]
        .expanding()
        .mean()
        .reset_index(level=[0, 1], drop=True)
    ).fillna(0.0)

    games["pts_diff_avg_L5"] = (
        games.groupby(["teamId", "season"])["pts_diff_L1"]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    ).fillna(0.0)

    games["pts_diff_avg_L13"] = (
        games.groupby(["teamId", "season"])["pts_diff_L1"]
        .rolling(window=13, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    ).fillna(0.0)

    games["pts_diff_avg_L26"] = (
        games.groupby(["teamId", "season"])["pts_diff_L1"]
        .rolling(window=26, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    ).fillna(0.0)

    return_cols = [
        "gameId",
        "gameDate",
        "season",
        "teamId",
        "pts_diff",
        "pts_diff_avg",
        "pts_diff_avg_L5",
        "pts_diff_avg_L13",
        "pts_diff_avg_L26",
    ]
    return games[return_cols].copy()
