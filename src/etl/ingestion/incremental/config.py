"""Configuration utilities for incremental ingestion."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict

from src.config.paths import (
    DEFAULT_FEATURES_CONFIG_PATH,
    project_relpath,
    RAW_GAMES_PATH,
    RAW_INCREMENTAL_ARCHIVE_DIR,
    RAW_INCREMENTAL_DIR,
    REGULAR_SEASON_GAMES_PATH,
    TEAMS_HISTORIES_CONFERENCE_NBA_CSV_PATH,
)


class IncrementalIngestionConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    raw_incremental_dir: str = project_relpath(RAW_INCREMENTAL_DIR)
    archive_dir: str = project_relpath(RAW_INCREMENTAL_ARCHIVE_DIR)
    raw_games_path: str = project_relpath(RAW_GAMES_PATH)
    processed_games_path: str = project_relpath(REGULAR_SEASON_GAMES_PATH)
    teams_history_path: str = project_relpath(TEAMS_HISTORIES_CONFERENCE_NBA_CSV_PATH)
    feature_config_path: str = project_relpath(DEFAULT_FEATURES_CONFIG_PATH)
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
