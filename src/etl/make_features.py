import argparse

import pandas as pd


from src.config.paths import (
    REGULAR_SEASON_GAMES_PATH,
    TEAMS_CITIES_CONFERENCE_HISTORY_PROCESSED_PATH,
    REGULAR_SEASON_GAMES_FEATURES_PATH,
)


from src.etl.transformation.add_conference import add_conference
from src.etl.features.aggregator import create_features_tables, merge_features

from src.etl.utils.common import add_neutral_court_game_flag
from src.ml.config.loader import load_features_config

from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)
setup_logging(level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Build feature tables from game data")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/features.yaml",
        help="Path to features config YAML (default: configs/features.yaml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    regular_season_games = pd.read_csv(
        REGULAR_SEASON_GAMES_PATH, parse_dates=["gameDate"], low_memory=False
    )
    teams_history = pd.read_csv(
        TEAMS_CITIES_CONFERENCE_HISTORY_PROCESSED_PATH,
    )

    # Step 0: Load configuration
    feature_engineering_config = load_features_config(args.config)
    record_lags = feature_engineering_config.record_lags
    point_differential_lags = feature_engineering_config.point_differential_lags
    location_lags = feature_engineering_config.location_lags
    distances_lags = feature_engineering_config.distances_lags
    sos_lags = feature_engineering_config.sos_lags
    sos_adj_alpha = feature_engineering_config.sos_adj_alpha
    sos_adj_location_lags = feature_engineering_config.features.sos_adj_record.location_lags
    gds_lags = feature_engineering_config.gds_lags
    gds_location_lags = feature_engineering_config.gds_location_lags
    gds_beta = feature_engineering_config.gds_beta

    # Step 1: Add neutral court game flag and conference information
    print("\n[Step 1/4] Adding neutral court game flag and conference information...")
    games_with_conference = add_conference(regular_season_games, teams_history)
    games_with_conference = add_neutral_court_game_flag(
        games_with_conference, game_label_column="gameLabel", drop_label_column=True
    )
    print(f"✓ Added conference information to {len(games_with_conference)} games")

    # Step 2: Create feature tables
    print("\n[Step 2/4] Creating feature tables...")
    create_features_tables(
        games_with_conference,
        record_lags,
        point_differential_lags,
        location_lags,
        distances_lags,
        sos_lags,
        sos_adj_alpha=sos_adj_alpha,
        sos_adj_location_lags=sos_adj_location_lags,
        gds_lags=gds_lags,
        gds_location_lags=gds_location_lags,
        gds_beta=gds_beta,
    )
    print("✓ Created all feature tables")

    # Step 3: Merge features
    print("\n[Step 3/3] Merging features into final table...")
    final_features = merge_features(games_with_conference)

    final_features.to_csv(REGULAR_SEASON_GAMES_FEATURES_PATH, index=False)
    print(f"✓ Created final features table with {len(final_features)} rows")
    print(f"✓ Saved to: {REGULAR_SEASON_GAMES_FEATURES_PATH}")


if __name__ == "__main__":
    main()
