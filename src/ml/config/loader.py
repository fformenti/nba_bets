"""Configuration loading utilities."""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any

from src.ml.config.schema import ExperimentConfig, PredictionConfig

logger = logging.getLogger(__name__)


def load_yaml_config(config_path: Path | str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_path : Path | str
        Path to YAML configuration file

    Returns
    -------
    dict
        Configuration dictionary

    Raises
    ------
    FileNotFoundError
        If config file does not exist
    ValueError
        If config file cannot be parsed
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML config file {config_path}: {e}") from e
    except Exception as e:
        raise ValueError(f"Error reading config file {config_path}: {e}") from e

    if config is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    logger.info(f"Loaded configuration from {config_path}")
    return config


def load_prediction_config(config_path: Path) -> PredictionConfig:
    """Load prediction config from YAML."""
    with open(config_path, "r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    return PredictionConfig(**raw_config)


def load_experiment_config(config_path: Path | str) -> ExperimentConfig:
    """
    Load configuration from a YAML file into a validated ExperimentConfig.

    Parameters
    ----------
    config_path : Path | str
        Path to YAML configuration file

    Returns
    -------
    ExperimentConfig
        Validated experiment configuration
    """
    raw_config = load_yaml_config(config_path)
    return ExperimentConfig.model_validate(raw_config)


def get_nested_config(
    config: Dict[str, Any], key_path: str, default: Any = None
) -> Any:
    """
    Get a nested configuration value using dot notation.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    key_path : str
        Dot-separated path to the configuration value (e.g., 'model.hyperparameter_tuning.enabled')
    default : any, optional
        Default value if key path not found

    Returns
    -------
    any
        Configuration value or default

    Examples
    --------
    >>> config = {'model': {'name': 'rf', 'params': {'n_estimators': 100}}}
    >>> get_nested_config(config, 'model.name')
    'rf'
    >>> get_nested_config(config, 'model.params.n_estimators')
    100
    >>> get_nested_config(config, 'model.params.max_depth', default=10)
    10
    """
    keys = key_path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def validate_config(config: Dict[str, Any], required_keys: list[str]) -> None:
    """
    Validate that required configuration keys are present.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    required_keys : list
        List of required key paths (dot notation)

    Raises
    ------
    ValueError
        If any required keys are missing
    """
    missing_keys = []

    for key_path in required_keys:
        if get_nested_config(config, key_path) is None:
            missing_keys.append(key_path)

    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
