"""Configuration schema for prediction pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


class PredictionConfig(BaseModel):
    """Configuration for running predictions on upcoming games."""

    model_config = ConfigDict(extra="ignore")

    input_dir: str = "data/raw/incremental/upcoming_games"
    output_path: str = "data/predictions/upcoming_games_predictions.csv"
    features_path: str = "data/processed/games_features.csv"
    feature_config_path: str = "configs/my_experiment.yaml"

    model_uri: str = Field(
        ...,
        description="MLflow model URI (e.g., 'models:/nba_classification_random_forest/Production' or 'runs:/<run_id>/model')",
    )

    tracking_uri: Optional[str] = None
    experiment_name: str = "nba_bets_predictions"

    conference_filter: str = Field(
        default="different",
        description="Filter games by conference matchup: different, same, or all.",
    )
    add_conference_delta: bool = False
    allow_missing_features: bool = True
    max_files: Optional[int] = None


def load_prediction_config(config_path: Path) -> PredictionConfig:
    """Load prediction config from YAML."""
    with open(config_path, "r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    return PredictionConfig(**raw_config)
