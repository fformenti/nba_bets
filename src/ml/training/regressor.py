"""Train regression models on a margin target.

Selected by ``data.target_column: margin`` (or ``margin_scaled``) in a training
config, through the same ``make train`` entry point as the classifiers. Kept
apart from ``experiment.train_single_model`` on purpose: nearly all of that
function is probability machinery — calibration, Brier, the market benchmark —
none of which applies to a predicted margin.

Two numbers per split: MAE, in the target's own units, and winner accuracy —
how often the predicted margin has the right sign — which is the one figure
directly comparable with a classifier's accuracy on the same games.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.ml.config.schema import ExperimentConfig
from src.ml.datasets.splits import build_splits
from src.ml.evaluation.metrics import compute_regression_metrics, get_model_display_name
from src.ml.features.engineering import identify_feature_types
from src.ml.features.preprocessing import create_preprocessing_pipeline
from src.ml.models.trainer import ModelTrainer
from src.ml.results import (
    RunInfo,
    build_predictions,
    git_commit,
    write_predictions,
    write_run_info,
)
from src.utils.logging_config import get_logger

from .runners import train_model_with_config
from .weighting import compute_sample_weights

logger = get_logger(__name__)


def winner_accuracy(y_true, y_pred) -> float:
    """Share of games where the predicted margin has the actual winner's sign.

    Margins are home minus away, so a positive prediction picks the home team.
    NBA games cannot end level, so ``y_true`` is never 0.
    """
    return float(np.mean((np.asarray(y_pred) > 0) == (np.asarray(y_true) > 0)))


def _split_metrics(trainer: ModelTrainer, X: pd.DataFrame, y: pd.Series, prefix: str) -> dict:
    metrics = trainer.evaluate(X, y, prefix=prefix)
    metrics[f"{prefix}_winner_accuracy"] = winner_accuracy(y, trainer.model.predict(X))
    return metrics


def _margin_series(
    run_id: str,
    model: str,
    split: str,
    splits,
    index: pd.Index,
    y_true: pd.Series,
    y_pred: np.ndarray,
    in_points: bool,
) -> pd.DataFrame:
    """One regressor's margins for ``split`` in the shared result schema.

    ``pred_margin`` is only filled when the target is the margin in points; a
    ``margin_scaled`` prediction is in other units, and writing it there would
    put two scales in one column. The winner call is valid either way: scaling
    is positive, so it never flips a sign.
    """
    metadata = splits.metadata.loc[index]
    true_margin = None
    if {"homeScore", "awayScore"} <= set(metadata.columns):
        true_margin = (
            pd.to_numeric(metadata["homeScore"], errors="coerce")
            - pd.to_numeric(metadata["awayScore"], errors="coerce")
        ).to_numpy()
    y_pred = np.asarray(y_pred, dtype=float)
    return build_predictions(
        run_id=run_id,
        model=model,
        split=split,
        game_ids=metadata["gameId"],
        y_true=(np.asarray(y_true.loc[index]) > 0).astype(int),
        y_pred=(y_pred > 0).astype(int),
        pred_margin=y_pred if in_points else None,
        true_margin=true_margin,
    )


def _summary(metrics: dict, prefix: str) -> str:
    return (
        f"MAE: {metrics[f'{prefix}_mae']:.3f} | RMSE: {metrics[f'{prefix}_rmse']:.3f} | "
        f"winner accuracy: {metrics[f'{prefix}_winner_accuracy'] * 100:.2f}%"
    )


def train_regression_model(
    config: ExperimentConfig,
    config_path: Optional[Path],
    tracker: Any,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Train every model in ``config.model.train_models`` as a regressor; ship the best on val MAE.

    Writes the same result files as the classifier, with ``pred_margin`` filled
    and ``p_home`` empty, so a regressor's winner calls plot next to a
    classifier's on the same games.
    """
    from src.ml.training.experiment import generate_run_name

    run_id = run_id or generate_run_name(config_path=config_path, include_timestamp=True)
    target = config.data.target_column
    if config.feature_selection.enabled:
        # Boruta-SHAP ranks features with an LGBMClassifier (src/ml/features/selection.py).
        raise ValueError(
            f"feature_selection is classifier-only; set feature_selection.enabled: false "
            f"to train on {target!r}."
        )

    tracker.log_params(
        {
            "target_column": target,
            "min_season": config.filters.min_season,
            "sample_weighting_enabled": config.weighting.enabled,
            "sample_weighting_K": config.weighting.saturation_K,
        }
    )
    tracker.set_tags(
        {
            "task": "regression",
            "config_file": config_path.stem if config_path else "default",
            "experiment_type": "training",
        }
    )
    if config.tags:
        tracker.set_tags(config.tags)
    if config.description:
        tracker.set_tags({"mlflow.note.content": config.description})

    splits = build_splits(config)
    X_train, X_val, X_test = splits.X_train, splits.X_val, splits.X_test
    y_train, y_val, y_test = splits.y_train, splits.y_val, splits.y_test
    logger.info(f"Data splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    tracker.log_params(
        {
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "n_features": len(X_train.columns),
            "split_method": splits.split_method,
            "test_start_season": config.splitting.test_start_season,
            "val_seasons": config.splitting.val_seasons,
        }
    )
    tracker.log_dict(
        {"features_used": sorted(X_train.columns.tolist())},
        artifact_file="feature_tracking/features_used.json",
    )

    sample_weight = compute_sample_weights(splits.metadata.loc[X_train.index], config.weighting)

    feature_types = identify_feature_types(X_train, exclude_columns=[])
    preprocessor = create_preprocessing_pipeline(
        numerical_features=feature_types["numerical"],
        categorical_features=feature_types["categorical"] or None,
        boolean_features=feature_types["boolean"] or None,
        scaling_method=config.preprocessing.scaling_method,
        imputation_strategy=config.preprocessing.imputation_strategy,
        handle_outliers=config.preprocessing.handle_outliers,
    )

    # The floor: predict the training mean margin — home advantage — for every game.
    mean_margin = np.full(len(y_test), float(y_train.mean()))
    baseline = {
        f"test_{k}": v for k, v in compute_regression_metrics(y_test, mean_margin).items()
    }
    baseline["test_winner_accuracy"] = winner_accuracy(y_test, mean_margin)
    logger.info(f"Mean-margin baseline — test {_summary(baseline, 'test')}")
    with tracker.child_run("mean_baseline"):
        tracker.log_metrics(baseline)
        tracker.set_tags({"model_type": "baseline"})

    in_points = target == "margin"
    prediction_frames = [
        _margin_series(
            run_id, "mean_baseline", "test", splits, X_test.index, y_test, mean_margin, in_points
        )
    ]

    results: Dict[str, Dict[str, Any]] = {}
    for model_name in config.model.train_models:
        logger.info(f"\n{'=' * 60}\nTraining {get_model_display_name(model_name)} regressor\n{'=' * 60}")
        try:
            pipeline, trainer, _ = train_model_with_config(
                model_name=model_name,
                model_config=config.model.model_dump(),
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                preprocessor=preprocessor,
                random_state=config.splitting.random_state,
                sample_weight=sample_weight,
                cv_strategy=config.model.hyperparameter_tuning.cv_strategy,
                task_type="regression",
            )
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}", exc_info=True)
            continue

        val_metrics = _split_metrics(trainer, X_val, y_val, "val")
        test_metrics = _split_metrics(trainer, X_test, y_test, "test")
        logger.info(f"Val:  {_summary(val_metrics, 'val')}")
        logger.info(f"Test: {_summary(test_metrics, 'test')}")
        results[model_name] = {
            "pipeline": pipeline,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }
        for split, X_split, y_split in (
            ("validation", X_val, y_val),
            ("test", X_test, y_test),
        ):
            prediction_frames.append(
                _margin_series(
                    run_id, model_name, split, splits, X_split.index, y_split,
                    trainer.model.predict(X_split), in_points,
                )
            )
        with tracker.child_run(model_name):
            tracker.log_metrics({**val_metrics, **test_metrics})
            tracker.set_tags({"model_type": model_name})

    if not results:
        return {}

    # Chosen on validation; test is reported, never selected on.
    best_model_name = min(results, key=lambda name: results[name]["val_metrics"]["val_mae"])
    best = results[best_model_name]
    logger.info(
        f"Best model: {get_model_display_name(best_model_name)} (selected on val MAE) — "
        f"test {_summary(best['test_metrics'], 'test')}"
    )
    tracker.log_metrics(best["test_metrics"])
    tracker.set_tags({"best_model": best_model_name})

    predictions = pd.concat(prediction_frames, ignore_index=True)
    for split in ("validation", "test"):
        write_predictions(predictions[predictions["split"] == split], run_id, split)
    write_run_info(
        RunInfo(
            run_id=run_id,
            family="sklearn",
            task="regression",
            selected_model=best_model_name,
            config_path=str(config_path) if config_path else None,
            mlflow_run_id=tracker.get_run_id(),
            mlflow_tracking_uri=config.mlflow.tracking_uri,
            git_commit=git_commit(),
            splits={
                "validation": {"n_games": len(X_val)},
                "test": {"n_games": len(X_test)},
            },
            extra={
                "description": config.description,
                "target_column": target,
                "n_features": len(X_train.columns),
                "split_method": splits.split_method,
            },
        )
    )

    artifact_path = f"nba_{best_model_name}"
    tracker.log_model(
        best["pipeline"],
        artifact_path=artifact_path,
        input_example=X_train.head(5),
        dataset_name=splits.date_range_name,
    )
    mlflow_run_id = tracker.get_run_id()
    run_model_uri = f"runs:/{mlflow_run_id}/{artifact_path}" if mlflow_run_id else None
    if run_model_uri:
        logger.info(f"Model logged at run URI: {run_model_uri}")
        tracker.set_tags({"model_run_uri": run_model_uri, "model_name": best_model_name})

    return {
        "best_model_name": best_model_name,
        "best_model_data": best,
        "models_to_train": results,
        "run_model_uri": run_model_uri,
        "run_id": run_id,
    }
