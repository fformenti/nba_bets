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

    # == Before 2024/25 ==
    games_before_2024_2025 = games[games["season"] <= "2023/24"]

    # Remove preseason, playoffs and play-in tournament from gameType
    gametype_exclude = ["Preseason", "Playoffs", "Play-in Tournament"]
    games_before_2024_2025 = games_before_2024_2025[
        ~games_before_2024_2025["gameType"].isin(gametype_exclude)
    ]

    # Remove NBA Cup Championship game
    games_before_2024_2025 = games_before_2024_2025[
        games_before_2024_2025["gameId"] != 62300001
    ]
    # Semifinals games Id (for location purposes)
    # semifinals_games_id = [22301230, 22301229]

    # == Before 2024/25 ==
    games_2024_2025 = games[games["season"] == "2024/25"]

    # Remove preseason, playoffs and play-in tournament from gameType
    gametype_exclude = ["Preseason", "Playoffs", "Play-in Tournament"]
    games_2024_2025 = games_2024_2025[
        ~games_2024_2025["gameType"].isin(gametype_exclude)
    ]

    # Remove Emirates NBA Cup Championship game
    gameSubLabel_exclude = ["Championship"]
    games_2024_2025 = games_2024_2025[
        ~games_2024_2025["gameSubLabel"].isin(gameSubLabel_exclude)
    ]

    # == From 2025/26  onwards ==
    games_after_2025 = games[games["season"] >= "2025/26"]

    gameType_exclude = ["Preseason", "Playoffs", "Play-in Tournament"]
    gameLabel_exclude = ["Preseason", "Playoffs", "Play-in Tournament"]
    gameSubLabel_exclude = ["Championship"]

    games_after_2025 = games_after_2025[
        ~games_after_2025["gameType"].isin(gameType_exclude)
    ]
    games_after_2025 = games_after_2025[
        ~games_after_2025["gameLabel"].isin(gameLabel_exclude)
    ]
    games_after_2025 = games_after_2025[
        ~games_after_2025["gameSubLabel"].isin(gameSubLabel_exclude)
    ]

    return pd.concat([games_before_2024_2025, games_2024_2025, games_after_2025])


def process_ingested_games() -> None:
    """Split ingested games into postponed and played regular-season sets."""
    ingested_games = pd.read_csv(
        INGESTED_GAMES_UPDATED_HISTORY_PATH, parse_dates=["gameDate"], low_memory=False
    )

    df_postponed = ingested_games[ingested_games["postponed"] == 1].copy()
    df_postponed.to_csv(POSTPONED_GAMES_PATH, index=False)

    parsed_played_games = ingested_games[ingested_games["postponed"] == 0]
    regular_season_games = filter_regular_season_games(parsed_played_games)
    regular_season_games.to_csv(REGULAR_SEASON_GAMES_PATH, index=False)
    logger.info(
        f"Saved {len(regular_season_games)} regular season games to {REGULAR_SEASON_GAMES_PATH}"
    )
