"""Winning percentage feature calculations."""

import pandas as pd


def calculate_record(games):
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

    games["record"] = (
        games.groupby(["teamId", "season"])["win_bool_l1"]
        .expanding()
        .mean()
        .round(2)
        .reset_index(level=[0, 1], drop=True)
    )

    games["record_L5"] = (
        games.groupby(["teamId", "season"])["win_bool_l1"]
        .rolling(window=5, min_periods=1)  # min_periods=1 to avoid NaN for small groups
        .mean()
        .round(2)
        .reset_index(level=[0, 1], drop=True)
    )

    games["record_L13"] = (
        games.groupby(["teamId", "season"])["win_bool_l1"]
        .rolling(window=13, min_periods=1)
        .mean()
        .round(2)
        .reset_index(level=[0, 1], drop=True)
    )

    games["record_L26"] = (
        games.groupby(["teamId", "season"])["win_bool_l1"]
        .rolling(window=26, min_periods=1)
        .mean()
        .round(2)
        .reset_index(level=[0, 1], drop=True)
    )

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
        "record",
        "record_L5",
        "record_L13",
        "record_L26",
    ]

    return games[return_cols].copy()


def calculate_home_record(home_games):
    """Calculate home team records."""
    home_games["win_bool"] = home_games.apply(
        lambda x: 1 if x["hometeamId"] == x["winner"] else 0, axis=1
    )

    home_games = home_games.rename(
        index=str,
        columns={
            "hometeamId": "teamId",
            "hometeamConference": "conference",
        },
    )

    return calculate_record(home_games)


def calculate_away_record(away_games):
    """Calculate away team records."""
    away_games["win_bool"] = away_games.apply(
        lambda x: 1 if x["winner"] != x["hometeamId"] else 0, axis=1
    )

    away_games = away_games.rename(
        index=str,
        columns={
            "awayteamId": "teamId",
            "awayteamConference": "conference",
        },
    )

    return calculate_record(away_games)


def make_east_west_record(games):
    """Calculate East vs West conference records."""
    east_west_record = games[games["hometeamConference"] != games["awayteamConference"]]
    east_west_record = east_west_record[
        ["gameDate", "gameDateOnlyStr", "season", "winnerteamConference"]
    ]
    east_west_record["east_winner"] = east_west_record["winnerteamConference"].apply(
        lambda x: 1 if x == "East" else -1
    )

    east_west_record["game_count_aux"] = east_west_record["east_winner"].abs()
    east_west_record_by_date = (
        east_west_record.groupby(["season", "gameDateOnlyStr"])
        .agg(
            east_wins=("east_winner", "sum"),
            games_played=("game_count_aux", "sum"),
        )
        .reset_index()
    )

    east_west_record_by_date["east_wins_cumsum"] = east_west_record_by_date.groupby(
        ["season"]
    )["east_wins"].cumsum()
    east_west_record_by_date["games_played_cumsum"] = east_west_record_by_date.groupby(
        ["season"]
    )["games_played"].cumsum()

    east_west_record_by_date["east_wins_pct"] = (
        east_west_record_by_date["east_wins_cumsum"]
        / east_west_record_by_date["games_played_cumsum"]
    ).round(2)

    east_west_record_by_date["east_wins_pct_L1"] = (
        east_west_record_by_date.groupby(["season"])["east_wins_pct"].shift(1).fillna(0)
    )

    return east_west_record_by_date
