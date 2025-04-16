import pandas as pd


def calculate_record(games):
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
        .reset_index(level=[0, 1], drop=True)
    )

    games["record_L5"] = (
        games.groupby(["teamId", "season"])["win_bool_l1"]
        .rolling(window=5, min_periods=1)  # min_periods=1 to avoid NaN for small groups
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    return_cols = [
        "gameId",
        "gameDate",
        "season",
        "teamId",
        "win_bool",
        "total_wins",
        "total_losses",
        "record",
        "record_L5",
    ]

    return games[return_cols].copy()


def calculate_home_record(home_games):
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


def calculate_pts_diff(games):
    games["pts_diff_l1"] = games.groupby(["teamId", "season"])["pts_diff"].shift(1)
    games["pts_diff_avg"] = (
        games.groupby(["teamId", "season"])["pts_diff_l1"]
        .expanding()
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    games["pts_diff_avg_L5"] = (
        games.groupby(["teamId", "season"])["pts_diff_l1"]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    return_cols = [
        "gameId",
        "season",
        "teamId",
        "pts_diff",
        "pts_diff_avg",
        "pts_diff_avg_L5",
    ]
    return games[return_cols].copy()


def calculate_home_pts_diff(games):
    games["pts_diff"] = games.apply(
        lambda x: x["homeScore"] - x["awayScore"],
        axis=1,
    )

    home_games = games.rename(
        index=str,
        columns={
            "hometeamId": "teamId",
        },
    )

    return calculate_pts_diff(home_games)


def calculate_away_pts_diff(games):
    games["pts_diff"] = games.apply(
        lambda x: x["awayScore"] - x["homeScore"],
        axis=1,
    )

    home_games = games.rename(
        index=str,
        columns={
            "awayteamId": "teamId",
        },
    )

    return calculate_pts_diff(home_games)


def make_east_west_record(games):
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
    )

    east_west_record_by_date["east_wins_pct_lag1"] = (
        east_west_record_by_date.groupby(["season"])["east_wins_pct"].shift(1).fillna(0)
    )

    return east_west_record_by_date


# def make_east_west_record(games, season_start, season_end):
#     date_range = pd.DataFrame(
#         {"dateTimeAux": pd.date_range(start=season_start, end=season_end, freq="D")}
#     )
#     date_range["gameDateOnlyStr"] = date_range["dateTimeAux"].dt.strftime("%Y-%m-%d")

#     east_west_record = games[games["hometeamConference"] != games["awayteamConference"]]
#     east_west_record = east_west_record[
#         ["gameDate", "gameDateOnlyStr", "winnerteamConference"]
#     ]
#     east_west_record["east_winner"] = east_west_record["winnerteamConference"].apply(
#         lambda x: 1 if x == "East" else -1
#     )

#     east_west_record = date_range.merge(
#         east_west_record,
#         how="left",
#         on="gameDateOnlyStr",
#     ).drop(columns=["dateTimeAux", "gameDate"])
#     east_west_record["east_winner"] = east_west_record["east_winner"].fillna(0)

#     east_west_record["game_count_aux"] = east_west_record["east_winner"].abs()

#     east_west_record["winnerteamConference"] = east_west_record[
#         "winnerteamConference"
#     ].fillna("")

#     east_west_record_by_date = (
#         east_west_record.groupby(["gameDateOnlyStr"])
#         .agg(
#             east_wins=("east_winner", "sum"),
#             games_played=("game_count_aux", "sum"),
#         )
#         .reset_index()
#     )

#     east_west_record_by_date["east_wins_cumsum"] = east_west_record_by_date[
#         "east_wins"
#     ].cumsum()

#     east_west_record_by_date["games_played_cumsum"] = east_west_record_by_date[
#         "games_played"
#     ].cumsum()

#     east_west_record_by_date["east_wins_pct"] = (
#         east_west_record_by_date["east_wins_cumsum"]
#         / east_west_record_by_date["games_played_cumsum"]
#     )

#     east_west_record_by_date["east_wins_pct_lag1"] = (
#         east_west_record_by_date["east_wins_pct"].shift(1).fillna(0)
#     )

#     return east_west_record_by_date
