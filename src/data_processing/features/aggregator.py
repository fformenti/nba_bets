"""Feature aggregation and merging utilities."""

import pandas as pd


from src.config import (
    LOCAL_REGULAR_SEASON_GAMES_PATH,
    LOCAL_GAMES_FEATURES_PATH,
    LOCAL_TEAMS_HISTORY_CITIES_CONFERENCES_PATH,
    LOCAL_TEAMS_HOME_RECORDS_PATH,
    LOCAL_TEAMS_AWAY_RECORDS_PATH,
    LOCAL_TEAMS_RECORDS_PATH,
    LOCAL_TEAMS_HOME_PTS_DIFF_PATH,
    LOCAL_TEAMS_AWAY_PTS_DIFF_PATH,
    LOCAL_TEAMS_PTS_DIFF_PATH,
    LOCAL_EAST_WEST_RECORDS_PATH,
    LOCAL_RESTED_DAYS_PATH,
)

from src.data_processing.features.winning_percentage import (
    calculate_away_record,
    calculate_home_record,
    calculate_record,
    make_east_west_record,
)
from src.data_processing.features.point_differential import (
    calculate_away_pts_diff,
    calculate_home_pts_diff,
    calculate_pts_diff,
)
from src.data_processing.features.rest_days import make_rested_days_table
from src.data_processing.transformation.add_teams_conferences import add_conference


def create_features_tables(games: pd.DataFrame):
    """
    Create all feature tables from games DataFrame.

    Parameters
    ----------
    games : pd.DataFrame
        Games DataFrame with conference information
    """
    home_records = calculate_home_record(games)
    away_records = calculate_away_record(games)
    teams_record = calculate_record(
        pd.concat([home_records, away_records], ignore_index=True)
    )

    games["pts_diff"] = games["homeScore"] - games["awayScore"]
    home_pts_diff = calculate_home_pts_diff(games)
    away_pts_diff = calculate_away_pts_diff(games)
    teams_pts_diff = calculate_pts_diff(pd.concat([home_pts_diff, away_pts_diff]))

    east_west_record = make_east_west_record(games)
    rested_days = make_rested_days_table(games)

    # Save the dataframes to CSV files
    home_records.to_csv(LOCAL_TEAMS_HOME_RECORDS_PATH, index=False)
    away_records.to_csv(LOCAL_TEAMS_AWAY_RECORDS_PATH, index=False)
    teams_record.to_csv(LOCAL_TEAMS_RECORDS_PATH, index=False)

    home_pts_diff.to_csv(LOCAL_TEAMS_HOME_PTS_DIFF_PATH, index=False)
    away_pts_diff.to_csv(LOCAL_TEAMS_AWAY_PTS_DIFF_PATH, index=False)
    teams_pts_diff.to_csv(LOCAL_TEAMS_PTS_DIFF_PATH, index=False)

    east_west_record.to_csv(LOCAL_EAST_WEST_RECORDS_PATH, index=False)
    rested_days.to_csv(LOCAL_RESTED_DAYS_PATH, index=False)
    return


def merge_features(games):
    """
    Merge all feature tables into the games DataFrame.

    This function replicates the logic from 03_merge_tables.py

    Parameters
    ----------
    games : pd.DataFrame
        Base games DataFrame

    Returns
    -------
    pd.DataFrame
        Games DataFrame with all features merged
    """
    # Load feature tables
    teams_home_record = pd.read_csv(LOCAL_TEAMS_HOME_RECORDS_PATH)
    teams_away_record = pd.read_csv(LOCAL_TEAMS_AWAY_RECORDS_PATH)
    teams_records = pd.read_csv(LOCAL_TEAMS_RECORDS_PATH)

    home_pts_diff = pd.read_csv(LOCAL_TEAMS_HOME_PTS_DIFF_PATH)
    away_pts_diff = pd.read_csv(LOCAL_TEAMS_AWAY_PTS_DIFF_PATH)
    teams_pts_diff = pd.read_csv(LOCAL_TEAMS_PTS_DIFF_PATH)

    east_west_record = pd.read_csv(LOCAL_EAST_WEST_RECORDS_PATH)
    rested_days = pd.read_csv(LOCAL_RESTED_DAYS_PATH)

    # Join Tables
    # Records
    teams_records = teams_records.drop(columns=["gameDate", "season", "win_bool"])
    games = _get_home_team_record(games, teams_records)
    games = _get_away_team_record(games, teams_records)

    teams_home_record = teams_home_record.drop(
        columns=["win_bool", "season", "gameDate"]
    )
    games = _get_home_team_record_at_home(games, teams_home_record)

    teams_away_record = teams_away_record.drop(
        columns=["win_bool", "season", "gameDate"]
    )
    games = _get_away_team_records_on_road(games, teams_away_record)

    # East vs West
    games = games.merge(
        east_west_record[["gameDateOnlyStr", "east_wins_pct_L1"]],
        how="left",
        on=["gameDateOnlyStr"],
    )

    # Point differential
    teams_pts_diff.drop(columns=["pts_diff", "season", "gameDate"], inplace=True)
    games = _add_home_team_point_diff(games, teams_pts_diff)
    games = _add_away_team_point_diff(games, teams_pts_diff)

    home_pts_diff.drop(columns=["pts_diff", "season", "gameDate"], inplace=True)
    games = _add_home_team_point_diff_at_home(games, home_pts_diff)
    away_pts_diff.drop(columns=["pts_diff", "season", "gameDate"], inplace=True)
    games = _add_away_team_point_diff_on_road(games, away_pts_diff)

    # Rested days
    games = _get_rested_days_home_team(games, rested_days)
    games = _get_rested_days_away_team(games, rested_days)

    games = games.drop(
        columns=[
            "hometeamName",
            "awayteamCity",
            "attendance",
            "arenaId",
        ]
    ).copy()

    return games


# ==== Teams Records ====
def _get_home_team_record(games, teams_records):
    games = (
        games.merge(
            teams_records,
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
            "record_L13": "record_L13_HT",
            "record_L26": "record_L26_HT",
            "games_played": "games_played_HT",
        }
    )

    return games


def _get_away_team_record(games, teams_records):
    games = (
        games.merge(
            teams_records,
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
            "record_L13": "record_L13_VT",
            "record_L26": "record_L26_VT",
            "games_played": "games_played_VT",
        }
    )

    return games


def _get_home_team_record_at_home(games, home_games_record):
    games = (
        games.merge(
            home_games_record,
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
            "record_L13": "record_L13_HT_at_home",
            "record_L26": "record_L26_HT_at_home",
            "games_played": "games_played_HT_at_home",
        }
    )

    return games


def _get_away_team_records_on_road(games, away_games_record):
    games = (
        games.merge(
            away_games_record,
            how="left",
            left_on=["gameId", "awayteamId"],
            right_on=["gameId", "teamId"],
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
            "record_L13": "record_L13_VT_on_road",
            "record_L26": "record_L26_VT_on_road",
            "games_played": "games_played_VT_on_road",
        }
    )

    return games


# ==== Point Differential ====
def _add_home_team_point_diff(games, teams_pts_diff):
    games = (
        games.merge(
            teams_pts_diff,
            how="left",
            left_on=["gameId", "hometeamId"],
            right_on=["gameId", "teamId"],
        )
        .drop(columns=["teamId"])
        .copy()
    )

    games = games.rename(
        columns={
            "pts_diff_avg": "pts_diff_avg_HT",
            "pts_diff_avg_L5": "pts_diff_avg_L5_HT",
            "pts_diff_avg_L13": "pts_diff_avg_L13_HT",
            "pts_diff_avg_L26": "pts_diff_avg_L26_HT",
        }
    )

    return games


def _add_home_team_point_diff_at_home(games, home_pts_diff):
    games = (
        games.merge(
            home_pts_diff,
            how="left",
            left_on=["gameId", "hometeamId"],
            right_on=["gameId", "teamId"],
        )
        .drop(columns=["teamId"])
        .copy()
    )

    games = games.rename(
        columns={
            "pts_diff_avg": "pts_diff_avg_HT_at_home",
            "pts_diff_avg_L5": "pts_diff_avg_L5_HT_at_home",
            "pts_diff_avg_L13": "pts_diff_avg_L13_HT_at_home",
            "pts_diff_avg_L26": "pts_diff_avg_L26_HT_at_home",
        }
    )

    return games


def _add_away_team_point_diff(games, teams_pts_diff):
    games = (
        games.merge(
            teams_pts_diff,
            how="left",
            left_on=["gameId", "awayteamId"],
            right_on=["gameId", "teamId"],
        )
        .drop(columns=["teamId"])
        .copy()
    )

    games = games.rename(
        columns={
            "pts_diff_avg": "pts_diff_avg_VT",
            "pts_diff_avg_L5": "pts_diff_avg_L5_VT",
            "pts_diff_avg_L13": "pts_diff_avg_L13_VT",
            "pts_diff_avg_L26": "pts_diff_avg_L26_VT",
        }
    )

    return games


def _add_away_team_point_diff_on_road(games, away_pts_diff):
    games = (
        games.merge(
            away_pts_diff,
            how="left",
            left_on=["gameId", "awayteamId"],
            right_on=["gameId", "teamId"],
        )
        .drop(columns=["teamId"])
        .copy()
    )

    games = games.rename(
        columns={
            "pts_diff_avg": "pts_diff_avg_VT_on_road",
            "pts_diff_avg_L5": "pts_diff_avg_L5_VT_on_road",
            "pts_diff_avg_L13": "pts_diff_avg_L13_VT_on_road",
            "pts_diff_avg_L26": "pts_diff_avg_L26_VT_on_road",
        }
    )

    return games


# ==== Rested Days ====
def _get_rested_days_home_team(games, rested_days):
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


def _get_rested_days_away_team(games, rested_days):
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
    # Read Tables
    games = pd.read_csv(LOCAL_REGULAR_SEASON_GAMES_PATH)
    teams_cities_conferences = pd.read_csv(LOCAL_TEAMS_HISTORY_CITIES_CONFERENCES_PATH)
    games = add_conference(games, teams_cities_conferences)

    # Create feature tables
    create_features_tables(games)

    # Merge features
    games_with_features = merge_features(games)

    # Save Table
    games_with_features.to_csv(LOCAL_GAMES_FEATURES_PATH, index=False)
