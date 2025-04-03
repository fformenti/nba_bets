from pandas import DataFrame
import pandas as pd

from games import read_games_file, add_features_to_games, filter_games
from calendar_info import make_rested_days_table
from seasons_info import make_season_info


def make_training_set(games_filtered):
    # Get seasons infos
    season_info_df = make_season_info(games_filtered)
    season_info_list = season_info_df.to_dict(orient="records")

    # Get seasons tables
    training_set_seasons_list = []
    for season_info in season_info_list:
        start_date = season_info["min"]
        end_date = season_info["max"]
        teams_season = season_info["hometeamId"]

        rested_days = make_rested_days_table(
            games_filtered, start_date, end_date, teams_season
        )
        # pd.concat([games_filtered, rested_days], axis=1)
        training_set_seasons_list.append(rested_days)

    return pd.concat(training_set_seasons_list, axis=1)


def main():
    gameType = "Regular Season"
    start_date = "1980-07-01"

    games = read_games_file()
    games: DataFrame = add_features_to_games(games)
    games_filtered: DataFrame = filter_games(games, start_date, gameType)
    training_set = make_training_set(games_filtered)
