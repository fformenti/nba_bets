"""Pydantic schemas for experiment configuration."""

from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator

from src.config.constants import (
    DEFAULT_METADATA_COLUMNS,
    INTERMEDIATE_COLUMNS,
)

_DEFAULT_CLASSIFIER_MODEL_NAMES: tuple[str, ...] = (
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "lgbm",
)


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str | None = None
    target_column: str = "win_bool"
    date_column: str = "gameDate"
    drop_na: bool = True


class WeightingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    saturation_K: int = Field(
        default=30,
        description="Games-played saturation point for sample weights. "
        "Observations where min(games_played_HT, games_played_VT) >= K get weight 1.0.",
    )
    season_decay_enabled: bool = Field(
        default=False,
        description="Enable exponential decay weighting by season to downweight older seasons.",
    )
    season_decay_lambda: float = Field(
        default=0.1,
        description="Decay rate for cross-season weighting. Higher = faster decay. "
        "Weight = exp(-lambda * seasons_ago).",
    )


class FiltersConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    min_season: str | None = None
    minimum_games_train: int = 15
    minimum_games_test: int = 15
    conference_filter: str = Field(
        default="same",
        description="Conference filter type: 'same', 'different', or 'all'.",
    )


class SplittingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    method: str = "temporal"
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42


class MomentumPairConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    feature: str = Field(
        description="Feature prefix used in delta column names, e.g. 'pts_diff_avg' or 'record'."
    )
    short: int = Field(description="Short window lag.")
    long: int = Field(description="Long window lag.")


class FeatureGroupConfig(BaseModel):
    """Configuration for a single feature group."""

    model_config = ConfigDict(extra="ignore")

    lags: list[int] = Field(default_factory=list)
    location_lags: list[int] = Field(default_factory=list)
    delta: bool = True
    enabled: bool = True


class FeaturesMapConfig(BaseModel):
    """Map of all feature groups with their delta/drop/enabled settings."""

    model_config = ConfigDict(extra="ignore")

    record: FeatureGroupConfig = Field(default_factory=FeatureGroupConfig)
    point_differential: FeatureGroupConfig = Field(
        default_factory=lambda: FeatureGroupConfig(enabled=False)
    )
    norm_point_differential: FeatureGroupConfig = Field(default_factory=FeatureGroupConfig)
    sos: FeatureGroupConfig = Field(
        default_factory=lambda: FeatureGroupConfig(delta=False)
    )
    sos_adj_record: FeatureGroupConfig = Field(default_factory=FeatureGroupConfig)
    distance: FeatureGroupConfig = Field(default_factory=FeatureGroupConfig)
    rested_days: FeatureGroupConfig = Field(default_factory=FeatureGroupConfig)
    streak: FeatureGroupConfig = Field(default_factory=FeatureGroupConfig)
    last_season_record: FeatureGroupConfig = Field(default_factory=FeatureGroupConfig)
    adjusted_last_season_record: FeatureGroupConfig = Field(default_factory=FeatureGroupConfig)
    gds: FeatureGroupConfig = Field(
        default_factory=lambda: FeatureGroupConfig(enabled=False)
    )
    home_and_road: FeatureGroupConfig = Field(
        default_factory=lambda: FeatureGroupConfig(delta=False)
    )
    indifference_flag: FeatureGroupConfig = Field(
        default_factory=lambda: FeatureGroupConfig(delta=False, enabled=False)
    )
    neutral_court: FeatureGroupConfig = Field(
        default_factory=lambda: FeatureGroupConfig(delta=False, enabled=False)
    )


class FeatureEngineeringConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    selection_mode: Literal["inclusion", "exclusion"] = Field(
        default="exclusion",
        description="'inclusion': only declared features are kept. "
        "'exclusion': metadata/intermediate columns are dropped (legacy).",
    )
    sos_adj_alpha: float = Field(
        default=1.0,
        description="Exponent for SOS-adjusted record: adj = raw * (sos / league_avg_sos) ^ alpha.",
    )
    gds_beta: float = Field(
        default=0.10,
        description="Home/away adjustment factor for Game Difficulty Score.",
    )
    features: FeaturesMapConfig = Field(default_factory=FeaturesMapConfig)
    momentum_pairs: list[MomentumPairConfig] = Field(
        default_factory=list,
        description="Pairs of (feature, short_lag, long_lag) to replace with a momentum delta feature.",
    )
    metadata_columns: list[str] = Field(
        default_factory=lambda: DEFAULT_METADATA_COLUMNS
    )
    intermediate_columns: list[str] = Field(
        default_factory=lambda: INTERMEDIATE_COLUMNS
    )
    exclude_columns: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def migrate_flat_lags(cls, data: Any) -> Any:
        """Migrate old flat-field configs to the new grouped features structure.

        Old MLflow artifacts may have:
            record_lags: [1,3,5,...]
            point_differential_lags: [...]
        instead of the new ``features:`` map. This validator converts them.
        """
        if not isinstance(data, dict):
            return data
        # Only migrate if 'features' is absent and old flat fields are present
        if "features" in data:
            return data

        flat_field_map = {
            "record_lags": ("record", "lags"),
            "point_differential_lags": ("point_differential", "lags"),
            "location_lags": ("record", "location_lags"),
            "distances_lags": ("distance", "lags"),
            "sos_lags": ("sos", "lags"),
        }

        features: dict = {}
        migrated = False
        for old_key, (group, field) in flat_field_map.items():
            if old_key in data:
                features.setdefault(group, {})[field] = data.pop(old_key)
                migrated = True

        # Copy location_lags to point_differential as well (they shared it)
        if "record" in features and "location_lags" in features["record"]:
            features.setdefault("point_differential", {})["location_lags"] = features[
                "record"
            ]["location_lags"]

        if migrated:
            data["features"] = features
            # Migrate originally_enriched_columns → intermediate_columns
            if "originally_enriched_columns" in data:
                data.pop("originally_enriched_columns")

        return data

    # Backward-compatible property accessors for ETL code
    @property
    def record_lags(self) -> list[int]:
        return self.features.record.lags

    @property
    def point_differential_lags(self) -> list[int]:
        return self.features.point_differential.lags

    @property
    def norm_point_differential_lags(self) -> list[int]:
        return self.features.norm_point_differential.lags

    @property
    def location_lags(self) -> list[int]:
        return self.features.record.location_lags

    @property
    def distances_lags(self) -> list[int]:
        return self.features.distance.lags

    @property
    def sos_lags(self) -> list[int]:
        return self.features.sos.lags

    @property
    def gds_lags(self) -> list[int]:
        return self.features.gds.lags

    @property
    def gds_location_lags(self) -> list[int]:
        return self.features.gds.location_lags


class FeatureSelectionConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    method: str = "boruta_shap"
    n_iterations: int = 20
    significance_level: float = 0.05
    include_tentative: bool = True
    random_state: int = 42


class PreprocessingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    scaling_method: str = "standard"
    imputation_strategy: str = "mean"
    handle_outliers: bool = True


class HyperparameterTuningConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    method: str = "random"
    n_iter: int = 50
    cv_folds: int = 5
    scoring: str = "accuracy"
    cv_strategy: str = Field(
        default="timeseries",
        description="CV strategy: 'timeseries' (TimeSeriesSplit) or 'kfold' (standard k-fold).",
    )
    param_grid: Optional[Dict[str, list]] = None


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str = "classification"
    name: str = "nba_bets_classifier"
    train_models: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_CLASSIFIER_MODEL_NAMES),
        description="Classifier models to train after baselines in multi-model runs.",
    )
    hyperparameter_tuning: HyperparameterTuningConfig = HyperparameterTuningConfig()
    random_forest: dict[str, Any] = Field(default_factory=dict)
    gradient_boosting: dict[str, Any] = Field(default_factory=dict)
    xgboost: dict[str, Any] = Field(default_factory=dict)
    lgbm: dict[str, Any] = Field(default_factory=dict)


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    save_visualizations: bool = True
    output_dir: str = "outputs"


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_registry: str = "models"
    outputs: str = "outputs"
    save_local_models: bool = True


class MLflowConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    experiment_name: str = "nba_bets_classification"
    tracking_uri: str = "sqlite:///mlflow.db"
    register_model: bool = Field(
        default=True,
        description="Whether to register the model in MLflow Model Registry. "
        "Set to False for experimental runs (use run-based URIs instead).",
    )


class FeaturesConfig(BaseModel):
    """Thin wrapper used by ETL scripts that only need feature engineering params."""

    model_config = ConfigDict(extra="ignore")
    feature_engineering: FeatureEngineeringConfig = FeatureEngineeringConfig()


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    description: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    data: DataConfig = DataConfig()
    filters: FiltersConfig = FiltersConfig()
    weighting: WeightingConfig = WeightingConfig()
    splitting: SplittingConfig = SplittingConfig()
    feature_engineering: FeatureEngineeringConfig = FeatureEngineeringConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    feature_selection: FeatureSelectionConfig = FeatureSelectionConfig()
    model: ModelConfig = ModelConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    paths: PathsConfig = PathsConfig()
    mlflow: MLflowConfig = MLflowConfig()


class PredictionPathsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input_dir: str = "data/raw/incremental/upcoming_games"
    output: str = "data/predictions/upcoming_games_predictions.csv"
    features: str = "data/processed/games_features.csv"


class PredictionConfig(BaseModel):
    """Configuration for running predictions on upcoming games."""

    model_config = ConfigDict(extra="ignore")

    data: DataConfig = DataConfig()
    paths: PredictionPathsConfig = PredictionPathsConfig()
    feature_config_path: Optional[str] = None

    model_uris: dict[str, str] = Field(
        ...,
        description="MLflow model URIs keyed by conference filter type, e.g. "
        "{'same': 'models:/...', 'different': 'models:/...'}",
    )

    mlflow: MLflowConfig = MLflowConfig(experiment_name="nba_bets_predictions")

    allow_missing_features: bool = False
    max_files: Optional[int] = None


# ===== LLM fine-tuning (QLoRA SFT) =====


class HubConfig(BaseModel):
    """Hugging Face Hub destination for adapters and checkpoints."""

    model_config = ConfigDict(extra="ignore")

    user: str = "fformenti"
    project_name: str = "nba-bets"
    private: bool = True
    push_checkpoints: bool = Field(
        default=True,
        description="Push intermediate checkpoints to the Hub so a dropped "
        "connection can be resumed from.",
    )


class LLMDataConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    dataset_name: str = "fformenti/nba-bets"
    train_split: str = "train"
    eval_split: str = "validation"
    text_field: str = "text"
    max_sequence_length: int = 1024
    max_train_samples: Optional[int] = Field(
        default=None, description="Cap training rows; useful for smoke tests."
    )


class QuantizationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: Literal["4bit", "8bit", "none"] = "8bit"
    compute_dtype: Literal["bfloat16", "float16"] = "bfloat16"
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_use_double_quant: bool = True


class LoraSettings(BaseModel):
    """LoRA hyperparameters (named to avoid clashing with peft.LoraConfig)."""

    model_config = ConfigDict(extra="ignore")

    r: int = 32
    alpha: int = 64
    dropout: float = 0.1
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    bias: Literal["none", "all", "lora_only"] = "none"


class LLMTrainerConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    epochs: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    optimizer: str = "paged_adamw_32bit"
    weight_decay: float = 0.001
    max_grad_norm: float = 0.3
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    logging_steps: int = 50
    save_steps: int = 500
    save_total_limit: int = 10
    seed: int = 42
    train_on_completion_only: bool = Field(
        default=False,
        description="Mask the prompt so loss is computed only over the answer "
        "following completion_template.",
    )
    completion_template: str = (
        "The home team finished the game with a point differential of"
    )


class LLMEvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    split: str = "test"
    size: int = 200
    max_new_tokens: int = 8
    seed: int = 42


class WandbConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    project: str = "nba-bets"
    watch: str = "gradients"


class LLMMLflowConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    experiment_name: str = "nba_bets_llm"
    tracking_uri: str = "sqlite:///mlflow.db"


class LLMTrackingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    wandb: WandbConfig = WandbConfig()
    mlflow: LLMMLflowConfig = LLMMLflowConfig()


class LLMPathsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    output_dir: str = "outputs/llm"


class LLMTrainingConfig(BaseModel):
    """Configuration for QLoRA supervised fine-tuning of a causal LM."""

    model_config = ConfigDict(extra="ignore")

    description: str | None = None
    run_name: Optional[str] = Field(
        default=None,
        description="Pin a run name to resume it. When None a timestamped "
        "name is generated at launch.",
    )
    base_model: str = "meta-llama/Meta-Llama-3.1-8B"
    hub: HubConfig = HubConfig()
    data: LLMDataConfig = LLMDataConfig()
    quantization: QuantizationConfig = QuantizationConfig()
    lora: LoraSettings = LoraSettings()
    training: LLMTrainerConfig = LLMTrainerConfig()
    evaluation: LLMEvaluationConfig = LLMEvaluationConfig()
    tracking: LLMTrackingConfig = LLMTrackingConfig()
    paths: LLMPathsConfig = LLMPathsConfig()
