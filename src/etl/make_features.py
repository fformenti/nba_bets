import pandas as pd


from src.config.paths import (
    REGULAR_SEASON_GAMES_PATH,
    TEAMS_ARENA_PATH,
    TEAMS_CITIES_CONFERENCE_HISTORY_PROCESSED_PATH,
    REGULAR_SEASON_GAMES_FEATURES_PATH,
)


from src.etl.transformation.add_conference import add_conference
from src.etl.features.aggregator import (
    create_features_tables_from_config,
    merge_features,
)

from src.etl.utils.common import add_neutral_court_game_flag, save_feature_table
from src.etl.features.teams_arena import build_teams_arena, add_neutral_court
from src.ml.config.loader import load_features_config

from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)
setup_logging(level="INFO")


def build_features(config_path: str = "configs/features.yaml") -> None:
    """Build every feature table and merge them into games_features.csv."""

    regular_season_games = pd.read_csv(
        REGULAR_SEASON_GAMES_PATH, parse_dates=["gameDate"], low_memory=False
    )
    teams_history = pd.read_csv(
        TEAMS_CITIES_CONFERENCE_HISTORY_PROCESSED_PATH,
    )

    # Step 0: Load configuration
    features_config = load_features_config(config_path)

    # Step 1: Add neutral court game flag and conference information
    logger.info("\n[Step 1/4] Adding neutral court game flag and conference information...")
    games_with_conference = add_conference(regular_season_games, teams_history)
    games_with_conference = add_neutral_court_game_flag(
        games_with_conference, game_label_column="gameLabel", drop_label_column=True
    )
    teams_arena = build_teams_arena(games_with_conference)
    teams_arena.to_csv(TEAMS_ARENA_PATH, index=False)
    games_with_conference = add_neutral_court(games_with_conference, teams_arena)
    logger.info(f"✓ Added conference information to {len(games_with_conference)} games")

    # Step 2: Create feature tables
    logger.info("\n[Step 2/4] Creating feature tables...")
    create_features_tables_from_config(games_with_conference, features_config)
    logger.info("✓ Created all feature tables")

    # Step 3: Merge features
    logger.info("\n[Step 3/3] Merging features into final table...")
    final_features = merge_features(games_with_conference)

    save_feature_table(final_features, REGULAR_SEASON_GAMES_FEATURES_PATH)
    logger.info(f"✓ Created final features table with {len(final_features)} rows")
    logger.info(f"✓ Saved to: {REGULAR_SEASON_GAMES_FEATURES_PATH}")
