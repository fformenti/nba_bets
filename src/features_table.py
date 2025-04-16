import pandas as pd

from pandas import DataFrame
from nba_bets.src.games import read_games_file, filter_regular_season_games
from nba_bets.src.rest_between_games import make_rested_days_table
from teams_records import (
    calculate_record,
    calculate_away_record,
    calculate_home_record,
    make_east_west_record,
)
from utils import get_nba_season
from teams_cities_conferences import create_cities_conference

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)

TEAMS_HOME_RECORDS_PATH = "../data/processed/teams_home_record.csv"
TEAMS_AWAY_RECORDS_PATH = "../data/processed/teams_away_record.csv"
TEAMS_RECORDS_PATH = "../data/processed/teams_records.csv"
EAST_WEST_RECORDS_PATH = "../data/processed/east_west_record.csv"
RESTED_DAYS_PATH = "../data/processed/rested_days.csv"
GAMES_ADDED_FEATURES_PART1_PATH = "../data/processed/games_added_features_part1.csv"


def add_conference(games, cities_conferences):
    games["winnerteamCity"] = games.apply(
        lambda x: str(x["hometeamCity"])
        if x["winner"] == x["hometeamId"]
        else str(x["awayteamCity"])
        if x["winner"] == x["awayteamId"]
        else "",
        axis=1,
    )

    games = games.merge(
        cities_conferences,
        how="left",
        left_on=["hometeamCity", "season"],
        right_on=["teamCity", "season"],
    ).drop(columns=["teamCity", "ye_date"])

    games = games.rename(columns={"conference": "hometeamConference"}, inplace=False)

    games = games.merge(
        cities_conferences,
        how="left",
        left_on=["awayteamCity", "season"],
        right_on=["teamCity", "season"],
    ).drop(columns=["teamCity", "ye_date"])
    games = games.rename(columns={"conference": "awayteamConference"}, inplace=False)

    games = games.merge(
        cities_conferences,
        how="left",
        left_on=["winnerteamCity", "season"],
        right_on=["teamCity", "season"],
    ).drop(columns=["teamCity", "ye_date"])
    games = games.rename(columns={"conference": "winnerteamConference"}, inplace=False)

    return games


def add_features_to_games(games):
    games["gameDateOnlyStr"] = games["gameDate"].dt.strftime("%Y-%m-%d")

    games["season"] = games["gameDate"].apply(get_nba_season)
    cities_conferences = create_cities_conference()
    games = add_conference(games, cities_conferences)

    return games


def create_features_tables(games: DataFrame):
    home_records = calculate_home_record(games)
    away_records = calculate_away_record(games)
    teams_record = calculate_record(
        pd.concat([home_records, away_records], ignore_index=True)
    )
    east_west_record = make_east_west_record(games)
    rested_days = make_rested_days_table(games)

    # Save the dataframes to CSV files
    home_records.to_csv(TEAMS_HOME_RECORDS_PATH, index=False)
    away_records.to_csv(TEAMS_AWAY_RECORDS_PATH, index=False)
    teams_record.to_csv(TEAMS_RECORDS_PATH, index=False)
    east_west_record.to_csv(EAST_WEST_RECORDS_PATH, index=False)
    rested_days.to_csv(RESTED_DAYS_PATH, index=False)
    return


if __name__ == "__main__":
    # Test different cutting dates with DVC
    start_date = "2020-08-01"

    games = read_games_file()
    games_regular_season: DataFrame = filter_regular_season_games(games, start_date)
    games_regular_season = games_regular_season.drop(
        columns=["gameType", "gameSubLabel", "gameLabel", "seriesGameNumber"]
    ).copy()
    games_regular_season = add_features_to_games(games_regular_season)
    games_regular_season.to_csv(GAMES_ADDED_FEATURES_PART1_PATH, index=False)
    create_features_tables(games_regular_season)
