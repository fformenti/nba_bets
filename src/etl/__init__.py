"""
ETL module for NBA betting project.

This module provides utilities for:
- Data ingestion (loading and parsing raw data)
- Data transformation (cleaning and basic transformations)
- Feature engineering (creating derived features)
- Specialized utilities (LLM dataset creation, etc.)

Import directly from submodules, e.g.:
- from src.etl.ingestion.raw_games import build_regular_season_games
- from src.etl.transformation.add_conference import add_conference
- from src.etl.features.aggregator import create_features_tables
"""
