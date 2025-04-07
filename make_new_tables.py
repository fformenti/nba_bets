import pandas as pd

from pandas import DataFrame
from games import read_games_file, add_features_to_games, filter_regular_season_games
from calendar_info import make_rested_days_table
from teams_records import (
    calculate_record,
    calculate_away_record,
    calculate_home_record,
    make_east_west_record,
)

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)

TEAMS_HOME_RECORDS_PATH = "data/new_tables/teams_home_record.csv"
TEAMS_AWAY_RECORDS_PATH = "data/new_tables/teams_away_record.csv"
TEAMS_RECORDS_PATH = "data/new_tables/teams_records.csv"
EAST_WEST_RECORDS_PATH = "data/new_tables/east_west_record.csv"
RESTED_DAYS_PATH = "data/new_tables/rested_days.csv"
GAMES_ADDED_FEATURES_PART1_PATH = "data/new_tables/games_added_features_part1.csv"


def create_new_features_tables(games: DataFrame):
    seasons = games["season"].unique()
    home_games_record_season = []
    away_games_record_season = []
    teams_games_record_season = []
    east_west_record_season = []
    rested_days_season = []
    for season in seasons:
        games_season = games.loc[games["season"] == season].copy()
        season_start = games_season["gameDate"].min()
        season_end = games_season["gameDate"].max()
        season_teams_ids = games_season["hometeamId"].unique()

        # Home Records info
        home_games_record = calculate_home_record(games_season)
        home_games_record_season.append(home_games_record)

        # Away Records info
        away_games_record = calculate_away_record(games_season)
        away_games_record_season.append(away_games_record)

        # Team Records info
        teams_record = calculate_record(
            pd.concat([home_games_record, away_games_record], ignore_index=True)
        )
        teams_games_record_season.append(teams_record)

        # East vs West Record
        east_west_record_season.append(
            make_east_west_record(games, season_start, season_end)
        )

        # Days in Between Games
        rested_days_season.append(
            make_rested_days_table(
                games_season, season_start, season_end, season_teams_ids
            )
        )

    home_games_record = pd.concat(home_games_record_season, ignore_index=True)
    away_games_record = pd.concat(away_games_record_season, ignore_index=True)
    teams_games_record = pd.concat(teams_games_record_season, ignore_index=True)
    east_west_record = pd.concat(east_west_record_season, ignore_index=True)
    rested_days = pd.concat(rested_days_season, ignore_index=True)

    # Save the dataframes to CSV files
    home_games_record.to_csv(TEAMS_HOME_RECORDS_PATH, index=False)
    away_games_record.to_csv(TEAMS_AWAY_RECORDS_PATH, index=False)
    teams_games_record.to_csv(TEAMS_RECORDS_PATH, index=False)
    east_west_record.to_csv(EAST_WEST_RECORDS_PATH, index=False)
    rested_days.to_csv(RESTED_DAYS_PATH, index=False)
    return


if __name__ == "__main__":
    # Test different cutting dates with DVC
    start_date = "2020-08-01"

    games = read_games_file()
    games_regular_season: DataFrame = filter_regular_season_games(games, start_date)
    games_regular_season: DataFrame = add_features_to_games(games_regular_season)
    games_regular_season = games_regular_season.drop(
        columns=["gameType", "gameSubLabel", "gameLabel", "seriesGameNumber"]
    ).copy()

    games_regular_season.to_csv(GAMES_ADDED_FEATURES_PART1_PATH, index=False)
    create_new_features_tables(games_regular_season)
