import pandas as pd

from pandas import DataFrame
from games import read_games_file, filter_regular_season_games
from rest_beetwen_games import make_rested_days_table
from teams_records import (
    calculate_record,
    calculate_away_record,
    calculate_home_record,
    make_east_west_record,
)
from utils import get_nba_season
from teams_cities_conferences import get_cities_conference

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)

TEAMS_HOME_RECORDS_PATH = "data/processed/teams_home_record.csv"
TEAMS_AWAY_RECORDS_PATH = "data/processed/teams_away_record.csv"
TEAMS_RECORDS_PATH = "data/processed/teams_records.csv"
EAST_WEST_RECORDS_PATH = "data/processed/east_west_record.csv"
RESTED_DAYS_PATH = "data/processed/rested_days.csv"
GAMES_ADDED_FEATURES_PART1_PATH = "data/processed/games_added_features_part1.csv"


def add_features_to_games(games):
    games["season"] = games["gameDate"].apply(get_nba_season)

    games["winnerteamCity"] = games.apply(
        lambda x: str(x["hometeamCity"])
        if x["winner"] == x["hometeamId"]
        else str(x["awayteamCity"])
        if x["winner"] == x["awayteamId"]
        else "",
        axis=1,
    )

    games["gameDateOnlyStr"] = games["gameDate"].dt.strftime("%Y-%m-%d")

    games["pts_diff"] = games.apply(
        lambda x: x["homeScore"] - x["awayScore"],
        axis=1,
    )

    games["winner_home_bool"] = games.apply(
        lambda x: 1 if x["winner"] == x["hometeamId"] else 0, axis=1
    )

    games["winner_away_bool"] = games.apply(
        lambda x: 1 if x["winner"] != x["hometeamId"] else 0, axis=1
    )

    cities_conferences = get_cities_conference()

    games = games.merge(
        cities_conferences,
        how="left",
        left_on="hometeamCity",
        right_on="teamCity",
    ).drop(columns=["teamCity"])

    games = games.rename(columns={"Conference": "hometeamConference"}, inplace=False)

    games = games.merge(
        cities_conferences,
        how="left",
        left_on="awayteamCity",
        right_on="teamCity",
    ).drop(columns=["teamCity"])
    games = games.rename(columns={"Conference": "awayteamConference"}, inplace=False)

    games = games.merge(
        cities_conferences,
        how="left",
        left_on="winnerteamCity",
        right_on="teamCity",
    ).drop(columns=["teamCity"])
    games = games.rename(columns={"Conference": "winnerteamConference"}, inplace=False)

    return games


def create_features_tables(games: DataFrame):
    seasons = games["season"].unique()
    home_games_record_season = []
    away_games_record_season = []
    teams_games_record_season = []
    east_west_record_season = []
    rested_days_season = []

    games = add_features_to_games(games)
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
    games_regular_season = games_regular_season.drop(
        columns=["gameType", "gameSubLabel", "gameLabel", "seriesGameNumber"]
    ).copy()

    games_regular_season.to_csv(GAMES_ADDED_FEATURES_PART1_PATH, index=False)
    create_features_tables(games_regular_season)
