"""Configuration classes for ML experiments."""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict


@dataclass
class MLConfig:
    """Configuration for ML experiments."""

    # Data configuration
    data_path: str
    target_column: str
    feature_columns: Optional[List[str]] = None
    date_column: Optional[str] = None

    # Split configuration
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: Optional[int] = 42
    split_method: str = "temporal"  # 'random', 'temporal', 'stratified'
    stratify_column: Optional[str] = None

    # Feature engineering
    numerical_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    scaling_method: str = "standard"  # 'standard', 'robust', 'minmax'
    imputation_strategy: str = "mean"
    handle_outliers: bool = True

    # Model configuration
    model_type: str = "classification"  # 'regression' or 'classification'
    model_name: str = "model"
    hyperparameter_tuning: bool = False
    param_grid: Optional[Dict[str, Any]] = None
    cv_folds: int = 5

    # Evaluation
    metrics: Optional[List[str]] = None

    # Paths
    model_registry_path: str = "models"
    output_path: str = "outputs"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MLConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    def save(self, file_path: Path):
        """Save config to JSON file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, file_path: Path) -> "MLConfig":
        """Load config from JSON file."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(file_path, "r") as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)


def load_config(file_path: Path) -> MLConfig:
    """
    Load ML configuration from file.

    Parameters
    ----------
    file_path : Path
        Path to config JSON file

    Returns
    -------
    MLConfig
        Loaded configuration
    """
    return MLConfig.load(file_path)


def save_config(config: MLConfig, file_path: Path):
    """
    Save ML configuration to file.

    Parameters
    ----------
    config : MLConfig
        Configuration to save
    file_path : Path
        Path to save config JSON file
    """
    config.save(file_path)
