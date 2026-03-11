"""Winning percentage feature calculations."""

import pandas as pd


def calculate_record(games, lags=[]):
    """
    Calculate team records (winning percentage) with rolling windows.

    Parameters
    ----------
    games : pd.DataFrame
        Games DataFrame with teamId, gameDate, season, win_bool columns

    Returns
    -------
    pd.DataFrame
        DataFrame with record statistics
    """
    # Sort by Team and Game Date to perform cumulative calculations correctly
    games = games.sort_values(["teamId", "gameDate", "season"])
    games["win_bool_l1"] = games.groupby(["teamId", "season"])["win_bool"].shift(1)

    games["total_wins"] = games.groupby(["teamId", "season"])["win_bool_l1"].transform(
        pd.Series.cumsum
    )

    games["total_losses"] = (
        (1 - games["win_bool_l1"])
        .groupby([games["teamId"], games["season"]])
        .expanding()
        .sum()
        .reset_index(level=[0, 1], drop=True)
    )

    # games["record"] = (
    #     games.groupby(["teamId", "season"])["win_bool_l1"]
    #     .expanding()
    #     .mean()
    #     .round(2)
    #     .reset_index(level=[0, 1], drop=True)
    # )

    record_lags_cols = []
    for lag in lags:
        games[f"record_L{lag}"] = (
            games.groupby(["teamId", "season"])["win_bool_l1"]
            .rolling(window=lag, min_periods=1)
            .mean()
            .round(2)
            .reset_index(level=[0, 1], drop=True)
        )
        record_lags_cols.append(f"record_L{lag}")

    games["total_wins"] = games["total_wins"].fillna(0).astype(int)
    games["total_losses"] = games["total_losses"].fillna(0).astype(int)
    games["games_played"] = games["total_wins"] + games["total_losses"]

    return_cols = [
        "gameId",
        "gameDate",
        "season",
        "teamId",
        "win_bool",
        "total_wins",
        "total_losses",
        "games_played",
    ] + record_lags_cols

    return games[return_cols].copy()


def calculate_home_record(home_games, lags=[]):
    """Calculate home team records."""
    home_games["win_bool"] = home_games.apply(
        lambda x: 1 if x["hometeamId"] == x["winner"] else 0, axis=1
    )

    home_games = home_games.rename(
        columns={
            "hometeamId": "teamId",
            "hometeamConference": "conference",
        },
    )

    return calculate_record(home_games, lags)


def calculate_away_record(away_games, lags=[]):
    """Calculate away team records."""
    df = away_games.copy()
    df["win_bool"] = df.apply(
        lambda x: 1 if x["winner"] != x["hometeamId"] else 0, axis=1
    )

    df = df.rename(
        columns={
            "awayteamId": "teamId",
            "awayteamConference": "conference",
        },
    )

    return calculate_record(df, lags)
