"""Configuration utilities for incremental ingestion."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict


class IncrementalIngestionConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    raw_incremental_dir: str = "data/raw/incremental"
    archive_dir: str = "data/raw/incremental/archive"
    raw_games_path: str = "data/raw/historical/games/Games.csv"
    processed_games_path: str = "data/processed/regular_season/games.csv"
    teams_history_path: str = "data/raw/historical/TeamsHistoriesConferenceNBA.csv"
    feature_config_path: str = "configs/my_experiment.yaml"
    current_season_year: int = 2024
    update_features: bool = True
    max_files: Optional[int] = None
    use_llm: bool = False
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: int = 30
    dedupe_key: str = "gameId"


def load_incremental_config(config_path: Path) -> IncrementalIngestionConfig:
    with open(config_path, "r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    return IncrementalIngestionConfig(**raw_config)
