from pandas import DataFrame
import pandas as pd

from src.config.constants import MIN_SEASON, REGULAR_SEASON_GAME_TYPE
from src.config.paths import (
    REGULAR_SEASON_GAMES_PATH,
    INGESTED_GAMES_UPDATED_HISTORY_PATH,
)
from src.etl.utils.common import atomic_write_csv, game_type_from_id
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

    regular_season_games = pd.concat(
        [games_before_2024_2025, games_2024_2025, games_after_2025]
    )
    regular_season_games = _drop_non_regular_season_ids(regular_season_games)
    return regular_season_games[regular_season_games["season"] >= MIN_SEASON]


def _drop_non_regular_season_ids(games: DataFrame) -> DataFrame:
    """Last guard: keep only gameIds that encode a regular-season game.

    The era branches above filter on gameType, gameLabel and gameSubLabel, all
    of which are blank on the oldest historical rows. Rather than let a blank
    field mean "keep", the gameId itself is checked — it is the one field every
    row has. Ids too malformed to read are kept, so this can only ever remove a
    game it is certain about.
    """
    game_types = games["gameId"].map(game_type_from_id)
    excluded = game_types.notna() & (game_types != REGULAR_SEASON_GAME_TYPE)
    if excluded.any():
        logger.info(
            "Dropping %d non-regular-season game(s) by gameId: %s",
            excluded.sum(),
            game_types[excluded].value_counts().to_dict(),
        )
    return games[~excluded]


def process_ingested_games() -> None:
    """Filter the history table down to played regular-season games."""
    ingested_games = pd.read_csv(
        INGESTED_GAMES_UPDATED_HISTORY_PATH, parse_dates=["gameDate"], low_memory=False
    )

    parsed_played_games = ingested_games[ingested_games["postponed"] == 0]
    regular_season_games = filter_regular_season_games(parsed_played_games)
    # atomic_write_csv rather than to_csv: it creates the directory (a fresh
    # clone has no data/processed/regular_season, and this step used to be the
    # one place in the pipeline that assumed it existed) and it writes via a
    # temp file, so an interrupted run cannot leave a truncated table that
    # looks like a genuinely short one.
    atomic_write_csv(regular_season_games, REGULAR_SEASON_GAMES_PATH)
    logger.info(
        f"Saved {len(regular_season_games)} regular season games to {REGULAR_SEASON_GAMES_PATH}"
    )
