"""
Data processing pipeline orchestrator.

This module provides a high-level interface to run the complete data processing pipeline
from raw data to final features table.
"""

import pandas as pd
from pathlib import Path


from src.config import (
    LOCAL_RAW_GAMES_PATH,
    LOCAL_REGULAR_SEASON_GAMES_PATH,
    LOCAL_TEAMS_HISTORY_CITIES_CONFERENCES_PATH,
    LOCAL_GAMES_FEATURES_PATH,
)

from src.data_processing.ingestion import (
    create_teams_history_table,
    parse_raw_games,
    filter_regular_season_games,
    get_nba_season,
)
from src.data_processing.transformation import add_conference
from src.data_processing.features import create_features_tables, merge_features
from src.data_processing.utils import filter_games_by_date


def run_full_pipeline(
    raw_games_path: Path = None,
    teams_history_path: Path = None,
    output_path: Path = None,
    current_season_year: int = 2024,
    min_date: str = None,
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
        Path to teams history CSV. If None, uses default from constants.
    output_path : Path, optional
        Path to save final features table. If None, uses default from constants.
    current_season_year : int, default=2024
        Current season year for teams history processing
    min_date : str, optional
        Minimum date to filter games (format: "YYYY-MM-DD").
        Only games on or after this date will be included in the final output.
        If None, no date filtering is applied.
    """
    print("=" * 60)
    print("NBA Data Processing Pipeline")
    print("=" * 60)

    # Step 1: Create teams history table
    print("\n[Step 1/5] Creating teams history table...")
    if teams_history_path is None:
        teams_history_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "raw"
            / "historical"
            / "TeamsHistoriesConferenceNBA.csv"
        )

    teams_history = create_teams_history_table(
        input_file=str(teams_history_path),
        output_file=str(LOCAL_TEAMS_HISTORY_CITIES_CONFERENCES_PATH),
        current_season_year=current_season_year,
    )
    print(f"✓ Created teams history table with {len(teams_history)} rows")

    # Step 2: Parse and filter raw games
    print("\n[Step 2/5] Parsing and filtering raw games...")
    if raw_games_path is None:
        raw_games_path = LOCAL_RAW_GAMES_PATH

    raw_games = pd.read_csv(raw_games_path, parse_dates=["gameDate"])
    parsed_games = parse_raw_games(raw_games)
    parsed_games["season"] = parsed_games["gameDate"].apply(get_nba_season)
    regular_season_games = filter_regular_season_games(parsed_games)
    regular_season_games.to_csv(LOCAL_REGULAR_SEASON_GAMES_PATH, index=False)
    print(f"✓ Processed {len(regular_season_games)} regular season games")

    # Step 3: Add conference information
    print("\n[Step 3/5] Adding conference information...")
    games_with_conference = add_conference(regular_season_games, teams_history)
    print(f"✓ Added conference information to {len(games_with_conference)} games")

    # Step 4: Create feature tables
    print("\n[Step 4/5] Creating feature tables...")
    create_features_tables(games_with_conference)
    print("✓ Created all feature tables")

    # Step 5: Merge features
    print("\n[Step 5/6] Merging features into final table...")
    if output_path is None:
        output_path = LOCAL_GAMES_FEATURES_PATH

    final_features = merge_features(games_with_conference)

    # Step 6: Filter by date if provided
    if min_date is not None:
        print(f"\n[Step 6/6] Filtering games by minimum date ({min_date})...")
        final_features = filter_games_by_date(final_features, min_date)
        print(f"✓ Filtered to {len(final_features)} games")

    final_features.to_csv(output_path, index=False)
    print(f"✓ Created final features table with {len(final_features)} rows")
    print(f"✓ Saved to: {output_path}")

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)

    return final_features


if __name__ == "__main__":
    start_date = "1980-08-01"
    run_full_pipeline(min_date=start_date)
