import pandas as pd

from make_new_tables import (
    TEAMS_HOME_RECORDS_PATH,
    TEAMS_AWAY_RECORDS_PATH,
    TEAMS_RECORDS_PATH,
    EAST_WEST_RECORDS_PATH,
    RESTED_DAYS_PATH,
    GAMES_ADDED_FEATURES_PART1_PATH,
)


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

HOME_AWAY_RECORDS_COLUMNS_JOIN = [
    "gameId",
    "teamId",
    "record",
    "record_L5",
    "pts_diff_avg",
    "pts_diff_avg_L5",
]


# ==== Teams Records ====
def get_home_team_record(games, teams_record):
    games = (
        games.merge(
            teams_record[TEAMS_RECORDS_COLUMNS_JOIN],
            how="left",
            left_on=["gameId", "hometeamId"],
            right_on=["gameId", "teamId"],
        )
        .drop(columns=["teamId"])
        .copy()
    )

    games = games.rename(
        columns={
            "total_wins": "total_wins_HT",
            "total_losses": "total_losses_HT",
            "record": "record_HT",
            "record_L5": "record_L5_HT",
            "pts_diff_avg": "pts_diff_avg_HT",
            "pts_diff_avg_L5": "pts_diff_avg_L5_HT",
            "games_played": "games_played_HT",
        }
    )

    return games


def get_away_team_record(games, teams_record):
    games = (
        games.merge(
            teams_record[TEAMS_RECORDS_COLUMNS_JOIN],
            how="left",
            left_on=["gameId", "awayteamId"],
            right_on=["gameId", "teamId"],
        )
        .drop(columns=["teamId"])
        .copy()
    )

    games = games.rename(
        columns={
            "total_wins": "total_wins_VT",
            "total_losses": "total_losses_VT",
            "record": "record_VT",
            "record_L5": "record_L5_VT",
            "pts_diff_avg": "pts_diff_avg_VT",
            "pts_diff_avg_L5": "pts_diff_avg_L5_VT",
            "games_played": "games_played_VT",
        }
    )

    return games


def get_home_team_record_at_home(games, home_games_record):
    games = (
        games.merge(
            home_games_record[HOME_AWAY_RECORDS_COLUMNS_JOIN],
            how="left",
            left_on=["gameId", "hometeamId"],
            right_on=["gameId", "teamId"],
        )
        .drop(columns=["teamId"])
        .copy()
    )

    games = games.rename(
        columns={
            "total_wins": "total_wins_HT_at_home",
            "total_losses": "total_losses_HT_at_home",
            "record": "record_HT_at_home",
            "record_L5": "record_L5_HT_at_home",
            "pts_diff_avg": "pts_diff_avg_HT_at_home",
            "pts_diff_avg_L5": "pts_diff_avg_L5_HT_at_home",
            "games_played": "games_played_HT_at_home",
        }
    )

    return games


def get_away_team_records_on_road(games, away_games_record):
    games = (
        games.merge(
            away_games_record[HOME_AWAY_RECORDS_COLUMNS_JOIN],
            how="left",
            left_on=["gameId", "awayteamId"],
            right_on=["gameId", "teamId"],
            suffixes=("", "_VT_at_away"),
        )
        .drop(columns=["teamId"])
        .copy()
    )

    games = games.rename(
        columns={
            "total_wins": "total_wins_VT_on_road",
            "total_losses": "total_losses_VT_on_road",
            "record": "record_VT_on_road",
            "record_L5": "record_L5_VT_on_road",
            "pts_diff_avg": "pts_diff_avg_VT_on_road",
            "pts_diff_avg_L5": "pts_diff_avg_L5_VT_on_road",
            "games_played": "games_played_VT_on_road",
        }
    )

    return games


# ==== Rested Days ====
def get_rested_days_home_team(games, rested_days):
    rest_home_teams_columns = [
        "gameDateOnlyStr",
        "teamId",
        "rested_days",
        "days_at_home",
    ]
    games = games.merge(
        rested_days[rest_home_teams_columns],
        how="left",
        left_on=["gameDateOnlyStr", "hometeamId"],
        right_on=["gameDateOnlyStr", "teamId"],
    ).drop(columns=["teamId"])

    return games.rename(
        columns={
            "rested_days": "rested_days_HT",
        }
    )


def get_rested_days_away_team(games, rested_days):
    rest_away_teams_columns = [
        "gameDateOnlyStr",
        "teamId",
        "rested_days",
        "days_on_road",
    ]
    games = games.merge(
        rested_days[rest_away_teams_columns],
        how="left",
        left_on=["gameDateOnlyStr", "awayteamId"],
        right_on=["gameDateOnlyStr", "teamId"],
    ).drop(columns=["teamId"])

    return games.rename(
        columns={
            "rested_days": "rested_days_VT",
        }
    )


if __name__ == "__main__":
    games = pd.read_csv(GAMES_ADDED_FEATURES_PART1_PATH)
    teams_home_record = pd.read_csv(TEAMS_HOME_RECORDS_PATH)
    teams_away_record = pd.read_csv(TEAMS_AWAY_RECORDS_PATH)
    teams_records = pd.read_csv(TEAMS_RECORDS_PATH)
    east_west_record = pd.read_csv(EAST_WEST_RECORDS_PATH)
    rested_days = pd.read_csv(RESTED_DAYS_PATH)

    games = get_home_team_record(games, teams_records)
    games = get_away_team_record(games, teams_records)

    games = get_home_team_record_at_home(games, teams_home_record)
    games = get_away_team_records_on_road(games, teams_away_record)

    games = games.merge(
        east_west_record[["gameDateOnlyStr", "east_wins_pct_lag1"]],
        how="left",
        on=["gameDateOnlyStr"],
    )

    games = get_rested_days_home_team(games, rested_days)
    games = get_rested_days_away_team(games, rested_days)

    games = games.drop(
        columns=[
            "hometeamName",
            "awayteamCity",
            "attendance",
            "arenaId",
            "winner",
            "winnerteamCity",
            "gameDateOnlyStr",
            "winner_away_bool",
            "winnerteamConference",
            "gameDateOnlyStr",
        ]
    ).copy()

    games.to_csv("data/new_tables/games_added_features_part2.csv", index=False)
