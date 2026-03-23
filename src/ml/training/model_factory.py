from typing import Any, Dict, Optional

from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def clean_feature_names(feature_names: list) -> list:
    cleaned = []
    for name in feature_names:
        if "__" in name:
            parts = name.split("__", 1)
            cleaned.append(parts[1] if len(parts) == 2 else name)
        else:
            cleaned.append(name)
    return cleaned


def create_model(
    model_name: str,
    model_config: Dict[str, Any],
    task_type: str,
    random_state: Optional[int] = None,
    config_param_grid: Optional[Dict[str, list]] = None,
) -> tuple[BaseEstimator, Dict[str, Any]]:
    if model_name == "random_forest":
        rf_config = model_config.get("random_forest", {})
        model = RandomForestClassifier(
            n_estimators=rf_config.get("n_estimators", 100),
            max_depth=rf_config.get("max_depth", 10),
            min_samples_split=rf_config.get("min_samples_split", 20),
            min_samples_leaf=rf_config.get("min_samples_leaf", 15),
            max_features=rf_config.get("max_features", "sqrt"),
            max_samples=rf_config.get("max_samples", 0.7),
            class_weight=rf_config.get("class_weight", "balanced"),
            random_state=random_state,
            n_jobs=-1,
        )
        param_grid = {
            "model__n_estimators": [100, 200, 300, 500],
            "model__max_depth": [5, 7, 10],
            "model__min_samples_split": [20, 30, 50],
            "model__min_samples_leaf": [10, 15, 20, 30],
            "model__max_features": ["sqrt", "log2", 0.5],
            "model__max_samples": [0.6, 0.7, 0.8],
        }
    elif model_name == "gradient_boosting":
        gb_config = model_config.get("gradient_boosting", {})
        model = GradientBoostingClassifier(
            n_estimators=gb_config.get("n_estimators", 100),
            learning_rate=gb_config.get("learning_rate", 0.1),
            max_depth=gb_config.get("max_depth", 5),
            min_samples_split=gb_config.get("min_samples_split", 20),
            min_samples_leaf=gb_config.get("min_samples_leaf", 10),
            max_features=gb_config.get("max_features", "sqrt"),
            random_state=random_state,
        )
        param_grid = {
            "model__n_estimators": [50, 100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 4, 5, 6],
            "model__min_samples_split": [10, 20, 30, 50],
            "model__min_samples_leaf": [5, 10, 15, 20],
            "model__subsample": [0.7, 0.8, 0.9],
            "model__max_features": ["sqrt", "log2", 0.5],
        }
    elif model_name == "xgboost":
        xgb_config = model_config.get("xgboost", {})
        model = XGBClassifier(
            n_estimators=xgb_config.get("n_estimators", 300),
            learning_rate=xgb_config.get("learning_rate", 0.05),
            max_depth=xgb_config.get("max_depth", 5),
            subsample=xgb_config.get("subsample", 0.8),
            colsample_bytree=xgb_config.get("colsample_bytree", 0.8),
            reg_alpha=xgb_config.get("reg_alpha", 0.1),
            reg_lambda=xgb_config.get("reg_lambda", 1.0),
            scale_pos_weight=xgb_config.get("scale_pos_weight", 1),
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
        param_grid = {
            "model__n_estimators": [100, 200, 300, 500],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 4, 5, 6],
            "model__subsample": [0.7, 0.8, 0.9],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__reg_alpha": [0, 0.1, 0.5],
            "model__reg_lambda": [0.5, 1.0, 2.0],
        }
    elif model_name == "lgbm":
        lgbm_config = model_config.get("lgbm", {})
        model = LGBMClassifier(
            n_estimators=lgbm_config.get("n_estimators", 300),
            learning_rate=lgbm_config.get("learning_rate", 0.05),
            max_depth=lgbm_config.get("max_depth", 6),
            num_leaves=lgbm_config.get("num_leaves", 31),
            min_child_samples=lgbm_config.get("min_child_samples", 20),
            subsample=lgbm_config.get("subsample", 0.8),
            colsample_bytree=lgbm_config.get("colsample_bytree", 0.8),
            reg_alpha=lgbm_config.get("reg_alpha", 0.1),
            reg_lambda=lgbm_config.get("reg_lambda", 1.0),
            class_weight=lgbm_config.get("class_weight", "balanced"),
            verbose=lgbm_config.get("verbose", -1),
            random_state=random_state,
            n_jobs=-1,
        )
        param_grid = {
            "model__n_estimators": [100, 200, 300, 500],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [4, 6, 8],
            "model__num_leaves": [15, 23, 31],
            "model__min_child_samples": [20, 30, 50],
            "model__subsample": [0.7, 0.8, 0.9],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__reg_alpha": [0, 0.1, 0.5],
            "model__reg_lambda": [0.5, 1.0, 2.0],
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if config_param_grid:
        param_grid = {
            k if k.startswith("model__") else f"model__{k}": v
            for k, v in config_param_grid.items()
        }
        # logger.info(f"Using config-defined param grid: {param_grid}")

    logger.info(f"Created {model_name} model")
    return model, param_grid
