"""Configuration loading utilities."""

import copy
import logging
import yaml
from pathlib import Path
from typing import Dict, Any

from src.ml.config.schema import (
    ExperimentConfig,
    FeaturesConfig,
    LLMTrainingConfig,
    PredictionConfig,
)

logger = logging.getLogger(__name__)


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge *overlay* into *base*. Overlay wins for non-dict values."""
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_includes(raw_config: dict, config_dir: Path) -> dict:
    """Resolve ``_include`` directives by deep-merging base configs in order.

    Each path in the ``_include`` list is loaded and merged left-to-right,
    then the current file's own keys are merged on top.  Paths are resolved
    relative to *config_dir* (the directory of the file that contains the
    ``_include`` key).
    """
    includes = raw_config.pop("_include", None)
    if not includes:
        return raw_config

    merged: dict = {}
    for inc_path in includes:
        resolved = (config_dir / inc_path).resolve()
        inc_raw = load_yaml_config(resolved)
        # Recursively resolve nested includes
        inc_raw = _resolve_includes(inc_raw, resolved.parent)
        merged = _deep_merge(merged, inc_raw)

    # The current file's own keys take precedence over everything included
    merged = _deep_merge(merged, raw_config)
    return merged


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

    Supports ``_include`` directives for composing configs from multiple files.

    Parameters
    ----------
    config_path : Path | str
        Path to YAML configuration file

    Returns
    -------
    ExperimentConfig
        Validated experiment configuration
    """
    config_path = Path(config_path).resolve()
    raw_config = load_yaml_config(config_path)
    raw_config = _resolve_includes(raw_config, config_path.parent)
    return ExperimentConfig.model_validate(raw_config)


def load_llm_training_config(config_path: Path | str) -> LLMTrainingConfig:
    """
    Load an LLM fine-tuning config from a YAML file.

    Supports ``_include`` directives, same as :func:`load_experiment_config`.

    Parameters
    ----------
    config_path : Path | str
        Path to YAML configuration file (e.g. configs/train_llm/*.yaml)

    Returns
    -------
    LLMTrainingConfig
        Validated LLM training configuration
    """
    config_path = Path(config_path).resolve()
    raw_config = load_yaml_config(config_path)
    raw_config = _resolve_includes(raw_config, config_path.parent)
    return LLMTrainingConfig.model_validate(raw_config)


def load_features_config(config_path: Path | str):
    """Load only the feature engineering config from a YAML file.

    This is used by ETL scripts that only need lag/alpha parameters,
    not the full experiment configuration.
    """
    config_path = Path(config_path).resolve()
    raw_config = load_yaml_config(config_path)
    raw_config = _resolve_includes(raw_config, config_path.parent)
    return FeaturesConfig.model_validate(raw_config).feature_engineering


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
