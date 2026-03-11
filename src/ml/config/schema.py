"""Pydantic schemas for experiment configuration."""

from typing import Any, Optional
from pydantic import BaseModel, Field, ConfigDict

from src.config.constants import EARLIEST_GAME_DATE


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str | None = None
    target_column: str = "win_bool"
    date_column: str = "gameDate"
    drop_na: bool = True


class FiltersConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    start_date: str | None = EARLIEST_GAME_DATE
    minimum_games: int = 10
    conference_filter: str = Field(
        default="all",
        description="Conference filter type: 'same' (same conference only), "
        "'different' (different conferences only), or 'all' (all games). "
        "Determines which conference features to create.",
    )


class SplittingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    method: str = "temporal"
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42


class FeatureEngineeringConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    lags: list[int] = Field(default_factory=list)
    location_lags: list[int] = Field(default_factory=list)
    distances_lags: list[int] = Field(default_factory=list)
    metadata_columns: list[str] = Field(default_factory=list)
    originally_enriched_columns: list[str] = Field(default_factory=list)
    exclude_columns: list[str] = Field(default_factory=list)


class PreprocessingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    scaling_method: str = "standard"
    imputation_strategy: str = "mean"
    handle_outliers: bool = True


class HyperparameterTuningConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    method: str = "random"
    n_iter: int = 20
    cv_folds: int = 3
    scoring: str = "accuracy"


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str = "classification"
    name: str = "random_forest"
    register_model: bool = Field(
        default=True,
        description="Whether to register the model in MLflow Model Registry. "
        "Set to False for experimental runs (use run-based URIs instead).",
    )
    hyperparameter_tuning: HyperparameterTuningConfig = HyperparameterTuningConfig()
    random_forest: dict[str, Any] = Field(default_factory=dict)
    gradient_boosting: dict[str, Any] = Field(default_factory=dict)


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    save_visualizations: bool = True
    output_dir: str = "outputs"


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_registry: str = "models"
    outputs: str = "outputs"
    save_local_models: bool = True


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    data: DataConfig = DataConfig()
    filters: FiltersConfig = FiltersConfig()
    splitting: SplittingConfig = SplittingConfig()
    feature_engineering: FeatureEngineeringConfig = FeatureEngineeringConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    model: ModelConfig = ModelConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    paths: PathsConfig = PathsConfig()


class PredictionConfig(BaseModel):
    """Configuration for running predictions on upcoming games."""

    model_config = ConfigDict(extra="ignore")

    data: DataConfig = DataConfig()
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
        description="Filter games by conference matchup: different, same, or all. "
        "Note: This is used for filtering only. Feature engineering uses "
        "conference_filter from the experiment config.",
    )
    allow_missing_features: bool = False
    max_files: Optional[int] = None
