import pandas as pd
from pathlib import Path
import sys

# Add project root to path to allow imports
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from rest_between_games import make_rested_days_table  # noqa: E402
from winning_percentage import (  # noqa: E402
    calculate_away_record,
    calculate_home_record,
    calculate_record,
    make_east_west_record,
)
from pandas import DataFrame  # noqa: E402

from point_differential import (  # noqa: E402
    calculate_away_pts_diff,
    calculate_home_pts_diff,
    calculate_pts_diff,
)
from src.local.constants import (  # noqa: E402
    LOCAL_REGULAR_SEASON_GAMES_PATH,
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


def add_conference(games, cities_conferences):
    games = games.merge(
        cities_conferences[["teamId", "Conference", "season"]],
        how="left",
        left_on=["hometeamId", "season"],
        right_on=["teamId", "season"],
    ).drop(columns=["teamId"])

    games = games.rename(columns={"Conference": "hometeamConference"}, inplace=False)

    games = games.merge(
        cities_conferences[["teamId", "Conference", "season"]],
        how="left",
        left_on=["awayteamId", "season"],
        right_on=["teamId", "season"],
    ).drop(columns=["teamId"])
    games = games.rename(columns={"Conference": "awayteamConference"}, inplace=False)

    games = games.merge(
        cities_conferences[["teamId", "Conference", "season"]],
        how="left",
        left_on=["winner", "season"],
        right_on=["teamId", "season"],
    ).drop(columns=["teamId"])
    games = games.rename(columns={"Conference": "winnerteamConference"}, inplace=False)

    return games


def create_features_tables(games: DataFrame):
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


if __name__ == "__main__":
    regular_season_games = pd.read_csv(LOCAL_REGULAR_SEASON_GAMES_PATH)
    teams_cities_conferences = pd.read_csv(LOCAL_TEAMS_HISTORY_CITIES_CONFERENCES_PATH)
    regular_season_games = add_conference(
        regular_season_games, teams_cities_conferences
    )

    create_features_tables(regular_season_games)
