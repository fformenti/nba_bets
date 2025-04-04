import pandas as pd


def total_losses(window):
    # Exclusive OR (XOR) to invert the boolean values
    return sum([bool(i) ^ 1 for i in window])


def calculate_record(games):
    # Sort by Team and Game Date to perform cumulative calculations correctly
    games = games.sort_values(["teamId", "gameDate"])
    games["total_wins"] = (
        games.groupby("teamId")["win_bool"].transform(pd.Series.cumsum).shift(1)
    )

    games["total_losses"] = (
        games.groupby("teamId")["win_bool"]
        .expanding()
        .apply(total_losses, raw=True)
        .reset_index(level=0, drop=True)
    ).shift(1)

    # Record before game ----
    games["record"] = (
        games.groupby("teamId")["win_bool"]
        .expanding()  # Uses all available prior rows in the group
        .mean()
        .reset_index(level=0, drop=True)
    ).shift(1)

    games["record_L5"] = (
        games.groupby("teamId")["win_bool"]
        .rolling(window=5, min_periods=1)  # min_periods=1 to avoid NaN for small groups
        .mean()
        .reset_index(level=0, drop=True)
    ).shift(1)

    # Point differential -----
    games["pts_diff_avg"] = (
        games.groupby("teamId")["pts_diff"]
        .expanding()  # Uses all available prior rows in the group
        .mean()
        .reset_index(level=0, drop=True)
    ).shift(1)

    games["pts_diff_avg_L5"] = (
        games.groupby("teamId")["pts_diff"]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    ).shift(1)

    # Games Played
    games["games_played"] = games["total_wins"] + games["total_losses"]

    # games["games_played"] = (
    #     games.groupby("teamId")["teamId"]  # Group by label
    #     .expanding()  # Expanding window
    #     .count()  # Count non-NaN values
    #     .reset_index(level=0, drop=True)  # Align with original DataFrame
    # )

    return games


def calculate_home_record(games):
    home_games_cols = [
        "gameId",
        "gameDate",
        "gameDateOnlyStr",
        "hometeamId",
        "hometeamConference",
        "homeScore",
        "pts_diff",
        "winner_home_bool",
    ]

    home_games = games[home_games_cols]

    home_games = home_games.rename(
        index=str,
        columns={
            "hometeamId": "teamId",
            "hometeamConference": "conference",
            "homeScore": "score",
            "winner_home_bool": "win_bool",
        },
    )

    return calculate_record(home_games)


def calculate_away_record(games):
    away_game_cols = [
        "gameId",
        "gameDate",
        "gameDateOnlyStr",
        "awayteamId",
        "awayteamConference",
        "awayScore",
        "pts_diff",
        "winner_away_bool",
    ]

    away_games = games[away_game_cols]

    away_games = away_games.rename(
        index=str,
        columns={
            "awayteamId": "teamId",
            "awayteamConference": "conference",
            "awayScore": "score",
            "winner_away_bool": "win_bool",
        },
    )

    return calculate_record(away_games)


def generate_standings(teams_record):
    # make standings table
    teams_record["rank"] = (
        (
            teams_record.sort_values(
                ["record", "pts_diff_avg"], ascending=[False, False]
            )
            .groupby(["conference", "gameDateOnlyStr"])["record"]
            .rank(method="min", ascending=False)
        )
        .fillna(1)
        .astype(int)
    )
    return teams_record
