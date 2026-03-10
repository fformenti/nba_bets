"""
Data processing pipeline orchestrator.

This module provides a high-level interface to run the complete data processing pipeline
from raw data to final features table.
"""

import pandas as pd
from pathlib import Path


from src.config.paths import (
    RAW_GAMES_PATH,
    INGESTED_GAMES_UPDATED_HISTORY_PATH,
    REGULAR_SEASON_GAMES_PATH,
    TEAMS_CITIES_CONFERENCE_HISTORY_HANDMADE_PATH,
    TEAMS_CITIES_CONFERENCE_HISTORY_PROCESSED_PATH,
    REGULAR_SEASON_GAMES_FEATURES_PATH,
    PROJECT_ROOT,
    INGESTED_GAMES_PATH,
    POSTPONED_GAMES_PATH,
)
from src.config.constants import EARLIEST_GAME_DATE, CURRENT_SEASON_START_YEAR

from src.etl.ingestion.raw_games import parse_raw_games
from src.etl.process_ingested_games import filter_regular_season_games

from src.etl.ingestion.teams_history import (
    create_teams_history_table,
    load_teams_history_table,
)
from src.etl.transformation.add_conference import add_conference
from src.etl.features.aggregator import create_features_tables, merge_features

from src.etl.utils.common import filter_games_by_date, add_neutral_court_game_flag
from src.ml.config.loader import load_experiment_config


def run_full_pipeline(
    raw_games_path: Path = None,
    teams_history_path: Path = None,
    output_path: Path = None,
    current_season_start_year: int = None,
    earliest_date: str = None,
):
    """
    Run the complete data processing pipeline.

    Steps:
    1. Create teams history table
    2. Parse and filter raw games
    3. Add conference information
    4. Create feature tables
    5. Merge all features into final table
    6. Filter games by minimum date (if provided)

    Parameters
    ----------
    raw_games_path : Path, optional
        Path to raw games CSV. If None, uses default from constants.
    teams_history_path : Path, optional
        Path to the precomputed teams history CSV. If None, uses default from constants.
    output_path : Path, optional
        Path to save final features table. If None, uses default from constants.
    current_season_start_year
        Current season year for teams history processing
    earliest_date : str, optional
        Minimum date to filter games (format: "YYYY-MM-DD").
        Only games on or after this date will be included in the final output.
        If None, no date filtering is applied.
    """
    print("=" * 60)
    print("NBA Data Processing Pipeline")
    print("=" * 60)

    # Step 0: Load configuration
    config_path = PROJECT_ROOT / "configs" / "my_experiment.yaml"
    config = load_experiment_config(config_path)
    filters_config = config.filters
    feature_engineering_config = config.feature_engineering
    earliest_date = filters_config.start_date
    lags = feature_engineering_config.lags
    location_lags = feature_engineering_config.location_lags

    # Step 1: Load teams history table (create it if missing)
    print("\n[Step 1/5] Loading teams history table...")
    try:
        teams_history = load_teams_history_table(
            processed_file=str(teams_history_path)
            if teams_history_path is not None
            else None
        )
        print(f"✓ Loaded teams history table with {len(teams_history)} rows")
    except FileNotFoundError:
        print("Teams history table missing. Creating it now...")
        raw_teams_history_path = (
            teams_history_path
            if teams_history_path is not None
            else TEAMS_CITIES_CONFERENCE_HISTORY_HANDMADE_PATH
        )
        teams_history = create_teams_history_table(
            input_file=str(raw_teams_history_path),
            output_file=str(TEAMS_CITIES_CONFERENCE_HISTORY_PROCESSED_PATH),
            current_season_start_year=current_season_start_year,
        )
        print(f"✓ Created teams history table with {len(teams_history)} rows")

    # Step 2: Parse and filter raw games
    print("\n[Step 2/5] Parsing and filtering raw games...")
    if raw_games_path is None:
        raw_games_path = RAW_GAMES_PATH

    raw_games = pd.read_csv(raw_games_path, parse_dates=["gameDate"], low_memory=False)
    parsed_games = parse_raw_games(raw_games)
    parsed_games.to_csv(INGESTED_GAMES_PATH, index=False)

    # save postponed games
    df_postponed = parsed_games[parsed_games["postponed"] == 1].copy()
    df_postponed.to_csv(POSTPONED_GAMES_PATH, index=False)

    # save regular season games
    parsed_played_games = parsed_games[parsed_games["postponed"] == 0]
    regular_season_games = filter_regular_season_games(parsed_played_games)
    regular_season_games.to_csv(REGULAR_SEASON_GAMES_PATH, index=False)
    print(f"✓ Processed {len(regular_season_games)} regular season games")

    # Step 3: Add neutral court game flag and conference information
    print("\n[Step 3/5] Adding neutral court game flag and conference information...")
    games_with_conference = add_conference(regular_season_games, teams_history)
    games_with_conference = add_neutral_court_game_flag(
        games_with_conference, game_label_column="gameLabel", drop_label_column=True
    )
    print(f"✓ Added conference information to {len(games_with_conference)} games")

    # Step 4: Create feature tables
    print("\n[Step 4/5] Creating feature tables...")
    create_features_tables(games_with_conference, lags, location_lags)
    print("✓ Created all feature tables")

    # Step 5: Merge features
    print("\n[Step 5/6] Merging features into final table...")
    if output_path is None:
        output_path = REGULAR_SEASON_GAMES_FEATURES_PATH

    final_features = merge_features(games_with_conference)

    # Step 6: Filter by date if provided
    if earliest_date is not None:
        print(f"\n[Step 6/6] Filtering games by earliest date ({earliest_date})...")
        final_features = filter_games_by_date(final_features, earliest_date)
        print(f"✓ Filtered to {len(final_features)} games")

    final_features.to_csv(output_path, index=False)
    print(f"✓ Created final features table with {len(final_features)} rows")
    print(f"✓ Saved to: {output_path}")

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)

    return final_features


if __name__ == "__main__":
    earliest_date = EARLIEST_GAME_DATE
    current_season_start_year = CURRENT_SEASON_START_YEAR
    run_full_pipeline(
        raw_games_path=INGESTED_GAMES_UPDATED_HISTORY_PATH,
        earliest_date=earliest_date,
        current_season_start_year=current_season_start_year,
    )
