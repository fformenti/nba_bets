from pandas import DataFrame
import pandas as pd

from src.config.paths import (
    REGULAR_SEASON_GAMES_PATH,
    INGESTED_GAMES_UPDATED_HISTORY_PATH,
    POSTPONED_GAMES_PATH,
)
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)
setup_logging(level="INFO")


def filter_regular_season_games(games) -> DataFrame:
    """Filter games to only include regular season games."""
    # Changes were made to the raw games season 2025/26 has a different patttern from previous years
    not_gametype_regular_season = ["Playoffs", "Preseason", "Play-in Tournament"]
    games = games.loc[~games["gameType"].isin(not_gametype_regular_season)]
    gamelabel_preseason = ["Preseason"]
    games = games.loc[~games["gameType"].isin(gamelabel_preseason)]
    games = games.drop(columns=["gameSubLabel", "seriesGameNumber"]).copy()

    return games


if __name__ == "__main__":
    ingested_games = pd.read_csv(
        INGESTED_GAMES_UPDATED_HISTORY_PATH, parse_dates=["gameDate"], low_memory=False
    )

    # save postponed games
    df_postponed = ingested_games[ingested_games["postponed"] == 1].copy()
    df_postponed.to_csv(POSTPONED_GAMES_PATH, index=False)

    parsed_played_games = ingested_games[ingested_games["postponed"] == 0]
    regular_season_games = filter_regular_season_games(parsed_played_games)
    regular_season_games.to_csv(REGULAR_SEASON_GAMES_PATH, index=False)
    # add logger info
    logger.info(
        f"Saved {len(regular_season_games)} regular season games to {REGULAR_SEASON_GAMES_PATH}"
    )
