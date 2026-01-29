"""Data ingestion module for loading and parsing raw data."""

from .teams_history import create_teams_history_table, load_teams_history_table
from .raw_games import (
    parse_raw_games,
    filter_regular_season_games,
    build_regular_season_games,
    get_nba_season,
)
from .incremental import run_incremental_pipeline, IncrementalIngestionConfig
from .collectors import collect_upcoming_games

__all__ = [
    "create_teams_history_table",
    "load_teams_history_table",
    "parse_raw_games",
    "filter_regular_season_games",
    "build_regular_season_games",
    "get_nba_season",
    "run_incremental_pipeline",
    "IncrementalIngestionConfig",
    "collect_upcoming_games",
]
