"""
Data processing module for NBA betting project.

This module provides utilities for:
- Data ingestion (loading and parsing raw data)
- Data transformation (cleaning and basic transformations)
- Feature engineering (creating derived features)
- Specialized utilities (LLM dataset creation, etc.)
"""

from .ingestion import (
    create_teams_history_table,
    parse_raw_games,
    filter_regular_season_games,
    get_nba_season,
)
from .transformation import add_conference
from .features import (
    create_features_tables,
    merge_features,
    calculate_record,
    calculate_home_record,
    calculate_away_record,
    make_east_west_record,
    calculate_pts_diff,
    calculate_home_pts_diff,
    calculate_away_pts_diff,
    make_rested_days_table,
)
from .utils import (
    calculate_arena_occupation,
    get_nba_season as get_nba_season_util,
    filter_games_by_date,
)
from .pipeline import run_full_pipeline

__all__ = [
    # Ingestion
    "create_teams_history_table",
    "parse_raw_games",
    "filter_regular_season_games",
    "get_nba_season",
    # Transformation
    "add_conference",
    # Features
    "create_features_tables",
    "merge_features",
    "calculate_record",
    "calculate_home_record",
    "calculate_away_record",
    "make_east_west_record",
    "calculate_pts_diff",
    "calculate_home_pts_diff",
    "calculate_away_pts_diff",
    "make_rested_days_table",
    # Utils
    "calculate_arena_occupation",
    "get_nba_season_util",
    "filter_games_by_date",
    # Pipeline
    "run_full_pipeline",
]
