import pandas as pd


TEAMS_RECORDS_COLUMNS_JOIN = [
    "gameId",
    "teamId",
    "total_wins",
    "total_losses",
    "record",
    "record_L5",
    "pts_diff_avg",
    "pts_diff_avg_L5",
    "games_played",
]


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
        .expanding()
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
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    ).shift(1)

    games["pts_diff_avg_L5"] = (
        games.groupby("teamId")["pts_diff"]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    ).shift(1)

    games["games_played"] = games["total_wins"] + games["total_losses"]

    return games[TEAMS_RECORDS_COLUMNS_JOIN].copy()


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


def make_east_west_record(games, season_start, season_end):
    date_range = pd.DataFrame(
        {"dateTimeAux": pd.date_range(start=season_start, end=season_end, freq="D")}
    )
    date_range["gameDateOnlyStr"] = date_range["dateTimeAux"].dt.strftime("%Y-%m-%d")

    east_west_record = games[games["hometeamConference"] != games["awayteamConference"]]
    east_west_record = east_west_record[
        ["gameDate", "gameDateOnlyStr", "winnerteamConference"]
    ]
    east_west_record["east_winner"] = east_west_record["winnerteamConference"].apply(
        lambda x: 1 if x == "East" else -1
    )

    east_west_record = date_range.merge(
        east_west_record,
        how="left",
        on="gameDateOnlyStr",
    ).drop(columns=["dateTimeAux", "gameDate"])
    east_west_record["east_winner"] = east_west_record["east_winner"].fillna(0)

    east_west_record["game_count_aux"] = east_west_record["east_winner"].abs()

    east_west_record["winnerteamConference"] = east_west_record[
        "winnerteamConference"
    ].fillna("")

    east_west_record_by_date = (
        east_west_record.groupby(["gameDateOnlyStr"])
        .agg(
            east_wins=("east_winner", "sum"),
            games_played=("game_count_aux", "sum"),
        )
        .reset_index()
    )

    east_west_record_by_date["east_wins_cumsum"] = east_west_record_by_date[
        "east_wins"
    ].cumsum()

    east_west_record_by_date["games_played_cumsum"] = east_west_record_by_date[
        "games_played"
    ].cumsum()

    east_west_record_by_date["east_wins_pct"] = (
        east_west_record_by_date["east_wins_cumsum"]
        / east_west_record_by_date["games_played_cumsum"]
    )

    east_west_record_by_date["east_wins_pct_lag1"] = (
        east_west_record_by_date["east_wins_pct"].shift(1).fillna(0)
    )

    return east_west_record_by_date
