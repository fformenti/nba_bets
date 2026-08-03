import inspect
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from src.ml.models.trainer import ModelTrainer
from src.utils.logging_config import get_logger

from .model_factory import create_model

logger = get_logger(__name__)


def train_baseline_model(
    baseline_model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    try:
        baseline_model.fit(X_test, y_test)

        baseline_trainer = ModelTrainer(
            model=baseline_model, task_type="classification"
        )
        baseline_trainer.is_fitted = True

        baseline_test_metrics = baseline_trainer.evaluate(X_test, y_test)

        return {
            "pipeline": baseline_model,
            "trainer": baseline_trainer,
            "training_results": {},
            "test_metrics": baseline_test_metrics,
        }
    except Exception as e:
        logger.error(
            f"Error training {baseline_model.name_caption}: {e}", exc_info=True
        )


def train_model_with_config(
    model_name: str,
    model_config: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    preprocessor: Any,
    random_state: Optional[int] = None,
    sample_weight: Optional[np.ndarray] = None,
    cv_strategy: str = "timeseries",
) -> tuple[Pipeline, ModelTrainer, Dict[str, Any]]:
    per_model_hp = model_config.get(model_name, {}).get("hyperparameter_tuning", {})
    global_hp = model_config.get("hyperparameter_tuning", {})
    hp_config = per_model_hp if per_model_hp else global_hp
    config_param_grid = (
        hp_config.get("param_grid") if isinstance(hp_config, dict) else None
    )

    model, param_grid = create_model(
        model_name,
        model_config,
        task_type="classification",
        random_state=random_state,
        config_param_grid=config_param_grid,
    )

    if (
        sample_weight is not None
        and "sample_weight" not in inspect.signature(model.fit).parameters
    ):
        logger.warning(
            f"{model_name} does not support sample_weight; training unweighted."
        )
        sample_weight = None

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    trainer = ModelTrainer(
        model=pipeline, task_type="classification", random_state=random_state
    )

    tuning_performed = False
    if hp_config.get("enabled", False):
        logger.info(f"Starting hyperparameter tuning for {model_name}")
        trainer.hyperparameter_tuning(
            X_train=X_train,
            y_train=y_train,
            param_grid=param_grid,
            method=hp_config.get("method", "random"),
            cv=hp_config.get("cv_folds", 5),
            n_iter=hp_config.get("n_iter", 100),
            scoring=hp_config.get("scoring", "f1"),
            cv_strategy=hp_config.get("cv_strategy", cv_strategy),
            sample_weight=sample_weight,
        )
        pipeline = trainer.model
        tuning_performed = True

    if tuning_performed:
        # Model was manually refit with sample weights after HP tuning — evaluate
        logger.info(f"Evaluating tuned {model_name} (already fit by grid search)...")
        train_metrics = trainer.evaluate(X_train, y_train, prefix="train")
        val_metrics = trainer.evaluate(X_val, y_val, prefix="val")
        training_results = {"train": train_metrics, "val": val_metrics}

        logger.info(
            f"HP tuning best CV score ({hp_config.get('scoring')}): "
            f"{trainer.search_results_.best_score_:.4f} "
        )

        # Attach grid search metadata
        search = trainer.search_results_
        training_results["grid_search"] = {
            "best_params": search.best_params_,
            "best_cv_score": search.best_score_,
            "cv_results": pd.DataFrame(search.cv_results_),
            "method": hp_config.get("method", "random"),
            "scoring": hp_config.get("scoring", "f1"),
            "cv_folds": hp_config.get("cv_folds", 5),
            "cv_strategy": hp_config.get("cv_strategy", cv_strategy),
            "n_iter": hp_config.get("n_iter", 100),
        }
    else:
        logger.info(f"Training {model_name}...")
        training_results = trainer.train(
            X_train, y_train, X_val, y_val, sample_weight=sample_weight
        )

    return pipeline, trainer, training_results
