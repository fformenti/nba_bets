"""Pydantic schemas for experiment configuration."""

from typing import Any
from pydantic import BaseModel, Field, ConfigDict


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str | None = None
    target_column: str = "win_bool"
    date_column: str = "gameDate"
    drop_na: bool = True


class FiltersConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    start_date: str | None = "1980-08-01"
    minimum_games: int = 10


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
