"""Feature aggregation and merging utilities."""

import pandas as pd


from src.config.paths import (
    TEAMS_HOME_RECORDS_PATH,
    TEAMS_AWAY_RECORDS_PATH,
    TEAMS_RECORDS_PATH,
    TEAMS_HOME_PTS_DIFF_PATH,
    TEAMS_AWAY_PTS_DIFF_PATH,
    TEAMS_PTS_DIFF_PATH,
    EAST_WEST_RECORDS_PATH,
    EAST_WEST_RECORDS_AT_EAST_PATH,
    EAST_WEST_RECORDS_AT_WEST_PATH,
    RESTED_DAYS_PATH,
    TEAMS_DISTANCES_PATH,
    LAST_SEASON_RECORD_PATH,
    LAST_SEASON_HOME_RECORD_PATH,
    LAST_SEASON_AWAY_RECORD_PATH,
    TEAMS_STREAKS_PATH,
    TEAMS_SOS_PATH,
)

from src.etl.features.winning_percentage import (
    calculate_away_record,
    calculate_home_record,
    calculate_record,
)

from src.etl.features.east_vs_west import make_east_west_record

from src.etl.features.point_differential import (
    calculate_away_pts_diff,
    calculate_home_pts_diff,
    calculate_pts_diff,
)
from src.etl.features.rest_days import make_rested_days_table

from src.etl.features.distances import make_teams_distances_table_season
from src.etl.features.last_season_record import (
    create_last_season_record,
    create_last_season_home_record,
    create_last_season_away_record,
)
from src.etl.features.streaks import calculate_streak
from src.etl.features.strength_of_schedule import calculate_strength_of_schedule


def create_features_tables(
    games: pd.DataFrame, record_lags=[], point_differential_lags=[], location_lags=[], distances_lags=[], sos_lags=[]
):
    """
    Create all feature tables from games DataFrame.

    Parameters
    ----------
    games : pd.DataFrame
        Games DataFrame with conference information
    record_lags : list
        List of lags to create record features for
    point_differential_lags : list
        List of lags to create point differential features for
    location_lags : list
        List of lags to create location-specific record features for
    """
    home_records = calculate_home_record(games, location_lags)
    away_records = calculate_away_record(games, location_lags)
    teams_record = calculate_record(
        pd.concat([home_records, away_records], ignore_index=True), record_lags
    )

    # Create pts_diff column needed for point differential calculations
    games["pts_diff"] = games["homeScore"] - games["awayScore"]

    home_pts_diff = calculate_home_pts_diff(games, location_lags)
    away_pts_diff = calculate_away_pts_diff(games, location_lags)
    teams_pts_diff = calculate_pts_diff(pd.concat([home_pts_diff, away_pts_diff]), point_differential_lags)

    east_west_record = make_east_west_record(games)
    east_west_record_at_east = make_east_west_record(games, location="East")
    east_west_record_at_west = make_east_west_record(games, location="West")
    rested_days = make_rested_days_table(games)
    teams_distances = make_teams_distances_table_season(distances_lags, games=games)

    # Save the dataframes to CSV files
    home_records.to_csv(TEAMS_HOME_RECORDS_PATH, index=False)
    away_records.to_csv(TEAMS_AWAY_RECORDS_PATH, index=False)
    teams_record.to_csv(TEAMS_RECORDS_PATH, index=False)

    home_pts_diff.to_csv(TEAMS_HOME_PTS_DIFF_PATH, index=False)
    away_pts_diff.to_csv(TEAMS_AWAY_PTS_DIFF_PATH, index=False)
    teams_pts_diff.to_csv(TEAMS_PTS_DIFF_PATH, index=False)

    east_west_record.to_csv(EAST_WEST_RECORDS_PATH, index=False)
    east_west_record_at_east.to_csv(EAST_WEST_RECORDS_AT_EAST_PATH, index=False)
    east_west_record_at_west.to_csv(EAST_WEST_RECORDS_AT_WEST_PATH, index=False)
    rested_days.to_csv(RESTED_DAYS_PATH, index=False)
    teams_distances.to_csv(TEAMS_DISTANCES_PATH, index=False)

    last_season_record = create_last_season_record(games)
    last_season_record.to_csv(LAST_SEASON_RECORD_PATH, index=False)

    last_season_home_record = create_last_season_home_record(games)
    last_season_home_record.to_csv(LAST_SEASON_HOME_RECORD_PATH, index=False)

    last_season_away_record = create_last_season_away_record(games)
    last_season_away_record.to_csv(LAST_SEASON_AWAY_RECORD_PATH, index=False)

    teams_streaks = calculate_streak(games)
    teams_streaks.to_csv(TEAMS_STREAKS_PATH, index=False)

    if sos_lags:
        teams_sos = calculate_strength_of_schedule(games, lags=sos_lags)
        teams_sos.to_csv(TEAMS_SOS_PATH, index=False)

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
    teams_home_record = pd.read_csv(TEAMS_HOME_RECORDS_PATH)
    teams_away_record = pd.read_csv(TEAMS_AWAY_RECORDS_PATH)
    teams_records = pd.read_csv(TEAMS_RECORDS_PATH)

    home_pts_diff = pd.read_csv(TEAMS_HOME_PTS_DIFF_PATH)
    away_pts_diff = pd.read_csv(TEAMS_AWAY_PTS_DIFF_PATH)
    teams_pts_diff = pd.read_csv(TEAMS_PTS_DIFF_PATH)

    east_west_record = pd.read_csv(EAST_WEST_RECORDS_PATH)
    east_west_record_at_east = pd.read_csv(EAST_WEST_RECORDS_AT_EAST_PATH)
    east_west_record_at_west = pd.read_csv(EAST_WEST_RECORDS_AT_WEST_PATH)
    rested_days = pd.read_csv(RESTED_DAYS_PATH)
    teams_distances = pd.read_csv(TEAMS_DISTANCES_PATH)
    last_season_record = pd.read_csv(LAST_SEASON_RECORD_PATH)
    last_season_home_record = pd.read_csv(LAST_SEASON_HOME_RECORD_PATH)
    last_season_away_record = pd.read_csv(LAST_SEASON_AWAY_RECORD_PATH)
    teams_streaks = pd.read_csv(TEAMS_STREAKS_PATH)

    # Records
    # teams_records = teams_records.drop(columns=["gameDate", "season", "win_bool"])
    games = join_games_and_teams_feature(games, teams_records, "hometeamId", "HT")
    games = join_games_and_teams_feature(games, teams_records, "awayteamId", "VT")
    games = join_games_and_teams_feature(
        games, teams_home_record, "hometeamId", "HT_at_home"
    )
    games = join_games_and_teams_feature(
        games, teams_away_record, "awayteamId", "VT_on_road"
    )

    # Point differential
    games = join_games_and_teams_feature(games, teams_pts_diff, "hometeamId", "HT")
    games = join_games_and_teams_feature(games, teams_pts_diff, "awayteamId", "VT")
    games = join_games_and_teams_feature(
        games, home_pts_diff, "hometeamId", "HT_at_home"
    )
    games = join_games_and_teams_feature(
        games, away_pts_diff, "awayteamId", "VT_on_road"
    )

    # East vs West
    games = games.merge(
        east_west_record,
        how="left",
        on=["gameDateOnlyStr"],
    )
    games = games.merge(
        east_west_record_at_east,
        how="left",
        on=["gameDateOnlyStr"],
    )
    games = games.merge(
        east_west_record_at_west,
        how="left",
        on=["gameDateOnlyStr"],
    )

    # Rested days
    games = get_rested_days(games, rested_days, is_hometeam=True)
    games = get_rested_days(games, rested_days, is_hometeam=False)

    # Distances (joins on gameDateOnlyStr since teams_distances has one row per team/date)
    games = join_games_and_teams_feature(games, teams_distances, "hometeamId", "HT")
    games = join_games_and_teams_feature(games, teams_distances, "awayteamId", "VT")

    games = games.merge(
        last_season_record.rename(columns={"last_season_record": "last_season_record_HT"}),
        how="left",
        left_on=["season", "hometeamId"],
        right_on=["season", "teamId"],
    ).drop(columns=["teamId"])

    games = games.merge(
        last_season_record.rename(columns={"last_season_record": "last_season_record_VT"}),
        how="left",
        left_on=["season", "awayteamId"],
        right_on=["season", "teamId"],
    ).drop(columns=["teamId"])

    games = games.merge(
        last_season_home_record.rename(columns={"last_season_record": "last_season_record_HT_at_home"}),
        how="left",
        left_on=["season", "hometeamId"],
        right_on=["season", "teamId"],
    ).drop(columns=["teamId"])

    games = games.merge(
        last_season_away_record.rename(columns={"last_season_record": "last_season_record_VT_on_road"}),
        how="left",
        left_on=["season", "awayteamId"],
        right_on=["season", "teamId"],
    ).drop(columns=["teamId"])

    games = join_games_and_teams_feature(games, teams_streaks, "hometeamId", "HT")
    games = join_games_and_teams_feature(games, teams_streaks, "awayteamId", "VT")

    # Strength of Schedule
    if TEAMS_SOS_PATH.exists():
        teams_sos = pd.read_csv(TEAMS_SOS_PATH)
        games = join_games_and_teams_feature(games, teams_sos, "hometeamId", "HT")
        games = join_games_and_teams_feature(games, teams_sos, "awayteamId", "VT")

    games = games.drop(
        columns=[
            "attendance",
            "arenaId",
        ]
    ).copy()

    return games


# ==== Teams Features ====
def join_games_and_teams_feature(games, teams_feature, join_column, suffix):
    feature_specific_cols = set(teams_feature.columns).difference(games.columns)
    join_cols = ["gameId", "season", "teamId"]
    keep_cols = list(feature_specific_cols.union(set(join_cols)))
    games = (
        games.merge(
            teams_feature[keep_cols],
            how="left",
            left_on=["gameId", "season", join_column],
            right_on=["gameId", "season", "teamId"],
        )
        .drop(columns=["teamId"])
        .copy()
    )
    cols_to_rename_dict = {col: f"{col}_{suffix}" for col in feature_specific_cols}
    return games.rename(columns=cols_to_rename_dict)


# ==== Rested Days ====
def get_rested_days(games: pd.DataFrame, rested_days: pd.DataFrame, is_hometeam: bool):
    rest_columns = [
        "gameDateOnlyStr",
        "teamId",
        "rested_days",
    ]
    if is_hometeam:
        rest_columns.append("days_at_home")
        join_column = "hometeamId"
        suffix = "HT"
    else:
        rest_columns.append("days_on_road")
        join_column = "awayteamId"
        suffix = "VT"
    games = games.merge(
        rested_days[rest_columns],
        how="left",
        left_on=["gameDateOnlyStr", join_column],
        right_on=["gameDateOnlyStr", "teamId"],
    ).drop(columns=["teamId"])

    return games.rename(
        columns={
            "rested_days": f"rested_days_{suffix}",
        }
    )
