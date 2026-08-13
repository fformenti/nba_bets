from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config.paths import (
    REGULAR_SEASON_GAMES_FEATURES_PATH,
    TRAIN_DEFAULTS_CONFIG_PATH,
)
from src.ml.config.loader import load_yaml_config
from src.ml.config.schema import ExperimentConfig
from src.ml.datasets.splits import build_splits
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold

from src.ml.evaluation.metrics import (
    compute_brier_score,
    compute_classification_metrics,
    compute_ece,
    compute_ece_ratio,
    format_metrics_line,
    get_model_display_name,
    print_metrics_summary,
)
from src.ml.evaluation.analysis import compute_season_phase_metrics, generate_analysis
from src.ml.evaluation.visualization import (
    plot_boruta_shap_results,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_prediction_accuracy_by_bin,
    plot_roc_curve,
)
from src.ml.features.selection import run_feature_selection
from src.ml.features.engineering import identify_feature_types
from src.ml.features.preprocessing import create_preprocessing_pipeline
from src.ml.models.baseline import PointDifferentialBaseline, RecordDifferenceBaseline
from src.ml.models.registry import ModelRegistry
from src.utils.logging_config import get_logger

from .model_factory import clean_feature_names
from .runners import train_baseline_model, train_model_with_config
from .weighting import compute_sample_weights, effective_sample_size

logger = get_logger(__name__)

# Rows the validation split needs before its Brier comparison means anything.
# Below this, no calibrator is chosen — the alternative is choosing one on test,
# which is what this replaced.
MIN_CALIBRATION_PROBE_ROWS = 400

# Folds used to score a calibration method out-of-fold across all of validation.
CALIBRATION_PROBE_FOLDS = 5


def _oof_calibrated_proba(
    base_pipeline: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    method: Optional[str],
) -> np.ndarray:
    """Out-of-fold P(home win) on validation, for scoring a calibration method.

    Every validation row is predicted by a calibrator that never saw it, so the
    whole split scores the method instead of half of it. This replaced a single
    positional half-split, which spent 491 of 982 rows and fit the probe on the
    first half of a season to score it on the second — folding within-season
    drift into a decision that was already inside the noise.

    ``method=None`` gives the uncalibrated baseline, scored on the identical rows
    in the identical order so the comparison stays paired.
    """
    oof = np.empty(len(X_val), dtype=float)
    folds = KFold(n_splits=CALIBRATION_PROBE_FOLDS, shuffle=False)
    for fit_idx, score_idx in folds.split(X_val):
        X_score = X_val.iloc[score_idx]
        if method is None:
            oof[score_idx] = _positive_class_proba(base_pipeline, X_score)
            continue
        probe = CalibratedClassifierCV(base_pipeline, method=method, cv="prefit")
        probe.fit(X_val.iloc[fit_idx], y_val.iloc[fit_idx])
        oof[score_idx] = _positive_class_proba(probe, X_score)
    return oof


def _paired_brier_margin(
    y_true: pd.Series, proba_a: np.ndarray, proba_b: np.ndarray
) -> tuple:
    """(delta, se) for Brier(a) - Brier(b), paired row by row.

    Brier is a mean of per-row squared errors, so the difference between two
    models scored on the same rows is a mean of per-row differences and its
    standard error is exact — no bootstrap needed. Pairing matters: the two
    models' errors are highly correlated, so the unpaired SE would be far too
    wide to resolve the difference at all.
    """
    y = np.asarray(y_true, dtype=float)
    d = (np.asarray(proba_a) - y) ** 2 - (np.asarray(proba_b) - y) ** 2
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(d.size))


def _positive_class_proba(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    """P(home win) as a 1-D array, whatever shape predict_proba returns."""
    proba = estimator.predict_proba(X)
    return proba[:, 1] if proba.ndim > 1 else proba


def _extract_explicit_model_param_keys() -> dict[str, set[str]]:
    """Read `_defaults.yaml` and return explicit top-level param keys per model.

    Always the one shared base. This used to resolve `_defaults.yaml` as a
    sibling of the experiment config, which pinned train configs to a directory
    that happens to contain one, for no benefit — there is only ever one.
    """
    defaults_path = TRAIN_DEFAULTS_CONFIG_PATH
    try:
        raw_config = load_yaml_config(defaults_path)
    except Exception as exc:
        logger.warning(
            "Could not read %s for explicit model param logging: %s",
            defaults_path,
            exc,
        )
        return {}
    model_block = raw_config.get("model", {}) if isinstance(raw_config, dict) else {}
    if not isinstance(model_block, dict):
        return {}

    excluded_model_keys = {"type", "name", "train_models", "hyperparameter_tuning"}
    explicit_keys_by_model: dict[str, set[str]] = {}
    for model_name, model_entry in model_block.items():
        if model_name in excluded_model_keys or not isinstance(model_entry, dict):
            continue
        explicit_keys_by_model[model_name] = {
            key for key in model_entry.keys() if key != "hyperparameter_tuning"
        }

    return explicit_keys_by_model


def _get_explicit_model_params_for_logging(
    model_name: str,
    model_config: dict[str, Any],
    explicit_param_keys_by_model: dict[str, set[str]],
) -> dict[str, str]:
    """Filter model params to only keys explicitly declared in `_defaults.yaml`."""
    configured_model_params = model_config.get(model_name, {})
    if not isinstance(configured_model_params, dict):
        return {}

    allowed_keys = explicit_param_keys_by_model.get(model_name, set())
    return {
        key: str(configured_model_params[key])
        for key in allowed_keys
        if key in configured_model_params
    }


def generate_run_name(
    config_path: Optional[Path] = None,
    model_name: Optional[str] = None,
    include_timestamp: bool = True,
) -> str:
    parts = [config_path.stem if config_path else "default"]
    if model_name:
        parts.append(model_name)
    if include_timestamp:
        parts.append(datetime.now().strftime("%Y%m%d-%H%M%S"))
    return "-".join(parts)


def train_single_model(
    config: ExperimentConfig,
    config_path: Optional[Path],
    tracker: Any,
) -> Dict[str, Any]:
    data_config = config.data
    split_config = config.splitting
    model_config = config.model.model_dump()
    explicit_param_keys_by_model = _extract_explicit_model_param_keys()
    eval_config = config.evaluation
    paths_config = config.paths
    feat_eng_config = config.feature_engineering
    filters_config = config.filters
    weighting_config = config.weighting

    data_path = (
        Path(data_config.path)
        if data_config.path
        else Path(REGULAR_SEASON_GAMES_FEATURES_PATH)
    )
    target_column = data_config.target_column

    minimum_games_train = filters_config.minimum_games_train
    minimum_games_test = filters_config.minimum_games_test

    test_size = split_config.test_size
    val_size = split_config.val_size
    random_state = split_config.random_state

    min_season = filters_config.min_season

    tracker.log_params(
        {
            "target_column": target_column,
            "test_size": test_size,
            "val_size": val_size,
            "minimum_games_train": minimum_games_train,
            "minimum_games_test": minimum_games_test,
            "min_season": min_season,
            "scaling_method": config.preprocessing.scaling_method,
            "sample_weighting_enabled": weighting_config.enabled,
            "sample_weighting_K": weighting_config.saturation_K,
            "feature_selection_enabled": config.feature_selection.enabled,
            "feature_selection_include_tentative": config.feature_selection.include_tentative,
            "feature_engineering_sos_adj_alpha": feat_eng_config.sos_adj_alpha,
        }
    )
    tracker.set_tags(
        {
            "task": "classification",
            "config_file": config_path.stem if config_path else "default",
            "experiment_type": "training",
        }
    )
    if config.tags:
        tracker.set_tags(config.tags)
    if config.description:
        tracker.set_tags({"mlflow.note.content": config.description})

    splits = build_splits(config)

    games_enriched = splits.games_enriched
    metadata = splits.metadata
    y = splits.y
    X_train, X_val, X_test = splits.X_train, splits.X_val, splits.X_test
    y_train, y_val, y_test = splits.y_train, splits.y_val, splits.y_test
    X_test_baseline, y_test_baseline = splits.X_test_baseline, splits.y_test_baseline
    metadata_test = splits.metadata_test

    if splits.date_range_name:
        tracker.log_dataset(
            df=games_enriched,
            source=str(data_path),
            name=splits.date_range_name,
            targets=target_column,
            context="training",
        )

    # --- Sample weight computation ---
    # Within-season ramp only; see src/ml/training/weighting.py.
    train_sample_weight = compute_sample_weights(metadata.loc[X_train.index], weighting_config)
    if train_sample_weight is not None:
        tracker.log_metrics(
            {"train_effective_sample_size": effective_sample_size(train_sample_weight)}
        )

    # logger.info(
    #     f"Features BEFORE selection ({len(X_train.columns)}): {sorted(X_train.columns.tolist())}"
    # )

    # --- Feature Selection (Boruta-SHAP) ---
    boruta_selector = None
    if config.feature_selection.enabled:
        logger.info("Running Boruta-SHAP feature selection on training data...")
        selected_features, boruta_selector = run_feature_selection(
            X_train,
            y_train,
            config.feature_selection,
            random_state=split_config.random_state,
        )
        X_train = X_train[selected_features]
        X_val = X_val[selected_features]
        X_test = X_test[selected_features]

        tracker.log_params(
            {
                "feature_selection_method": config.feature_selection.method,
                "feature_selection_n_iterations": config.feature_selection.n_iterations,
                "feature_selection_n_confirmed": len(
                    boruta_selector.confirmed_features_
                ),
                "feature_selection_n_rejected": len(boruta_selector.rejected_features_),
                "feature_selection_n_tentative": len(
                    boruta_selector.tentative_features_
                ),
            }
        )

        # The counts alone do not say *which* features a run chose, and an
        # MLflow param value would truncate a 16-name list. Without this,
        # answering "did these two runs select the same features?" meant loading
        # both model artifacts and reading feature_names_in_.
        tracker.log_dict(
            {
                "include_tentative": config.feature_selection.include_tentative,
                "selected": list(selected_features),
                "confirmed": list(boruta_selector.confirmed_features_),
                "tentative": list(boruta_selector.tentative_features_),
                "rejected": list(boruta_selector.rejected_features_),
            },
            artifact_file="feature_tracking/selected_features.json",
        )

        # Save JSON artifact with full results
        output_dir = Path(paths_config.outputs)
        fs_dir = output_dir / "feature_selection"
        fs_dir.mkdir(parents=True, exist_ok=True)
        boruta_selector.save_json(fs_dir / "boruta_shap_results.json")

        # Generate visualization
        if eval_config.save_visualizations:
            viz_dir = output_dir / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            fig_boruta = plot_boruta_shap_results(
                boruta_selector,
                top_n=50,
                title="Boruta-SHAP Feature Selection",
            )
            fig_boruta.savefig(
                viz_dir / "boruta_shap_selection.png", dpi=300, bbox_inches="tight"
            )
            plt.close(fig_boruta)
            logger.info("Boruta-SHAP visualization saved.")

    # logger.info(f"Features sent to model: {list(X_train.columns)}")
    logger.info(
        f"Data splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )
    tracker.log_params(
        {
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "n_features": len(X_train.columns),
            # The splitter that actually ran, plus its season boundaries. Without
            # these, runs whose train/val/test populations differ completely look
            # identical in MLflow, and metrics get compared across them.
            "split_method": splits.split_method,
            "test_start_season": split_config.test_start_season,
            "val_seasons": split_config.val_seasons,
        }
    )
    tracker.log_dict(
        {"features_used": sorted(X_train.columns.tolist())},
        artifact_file="feature_tracking/features_used.json",
    )

    feature_types = identify_feature_types(X_train, exclude_columns=[])
    numerical_features = feature_types["numerical"]
    categorical_features = feature_types["categorical"]
    boolean_features = feature_types["boolean"]

    logger.info(
        f"Feature types identified: {len(numerical_features)} numerical, "
        f"{len(categorical_features)} categorical, {len(boolean_features)} boolean features"
    )

    preproc_config = config.preprocessing
    preprocessor = create_preprocessing_pipeline(
        numerical_features=numerical_features,
        categorical_features=categorical_features if categorical_features else None,
        boolean_features=boolean_features if boolean_features else None,
        scaling_method=preproc_config.scaling_method,
        imputation_strategy=preproc_config.imputation_strategy,
        handle_outliers=preproc_config.handle_outliers,
    )

    models_to_train = {}

    # ====== Record Difference Baseline ======
    logger.info(f"\n{'=' * 60}")
    logger.info("Record Difference Baseline")
    logger.info(f"{'=' * 60}")

    baseline_model = RecordDifferenceBaseline()
    models_to_train[baseline_model.name] = train_baseline_model(
        baseline_model,
        X_test_baseline,
        y_test_baseline,
    )

    if baseline_model.name in models_to_train and models_to_train[baseline_model.name]:
        baseline_metrics = models_to_train[baseline_model.name].get("test_metrics", {})
        with tracker.child_run(baseline_model.name):
            tracker.log_metrics(baseline_metrics)
            tracker.set_tags({"model_type": "baseline"})

    # ====== Point Differential Baseline ======
    # This was commented out behind "TODO: make this feature available even when
    # it's not passed to train config". splits.py already solves that: it builds
    # pts_diff_avg_L82_delta onto X_test_baseline from the untouched feature
    # frame, so the column is there whether or not the experiment config enables
    # the point_differential group. The commented body also called an older API
    # (baseline_trainer, baseline_test_metrics) that no longer exists; this
    # mirrors the record baseline above instead.
    logger.info(f"\n{'=' * 60}")
    logger.info("Point Differential Baseline")
    logger.info(f"{'=' * 60}")

    baseline_model = PointDifferentialBaseline(feature_column="pts_diff_avg_L82_delta")
    models_to_train[baseline_model.name] = train_baseline_model(
        baseline_model,
        X_test_baseline,
        y_test_baseline,
    )

    if baseline_model.name in models_to_train and models_to_train[baseline_model.name]:
        baseline_metrics = models_to_train[baseline_model.name].get("test_metrics", {})
        with tracker.child_run(baseline_model.name):
            tracker.log_metrics(baseline_metrics)
            tracker.set_tags({"model_type": "baseline"})

    # dict.fromkeys dedups while preserving order: a name listed in both the
    # seeded baselines and config.model.train_models would otherwise be trained
    # twice, which a multi-family sweep makes easy to do by accident.
    model_names = list(
        dict.fromkeys(list(models_to_train.keys()) + list(config.model.train_models))
    )
    for model_name in model_names:
        if "baseline" in model_name:
            continue
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training {get_model_display_name(model_name)}")
        logger.info(f"{'=' * 60}")

        try:
            pipeline, trainer, training_results = train_model_with_config(
                model_name=model_name,
                model_config=model_config,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                preprocessor=preprocessor,
                random_state=random_state,
                sample_weight=train_sample_weight,
                cv_strategy=config.model.hyperparameter_tuning.cv_strategy,
            )

            test_metrics = trainer.evaluate(X_test, y_test)
            # Both branches of train_model_with_config put validation metrics
            # under "val". They used to be computed and dropped; model choice
            # and calibration choice are made on them, so they get logged.
            val_metrics = training_results.get("val", {})
            models_to_train[model_name] = {
                "pipeline": pipeline,
                "trainer": trainer,
                "training_results": training_results,
                "test_metrics": test_metrics,
                "val_metrics": val_metrics,
            }

            with tracker.child_run(model_name):
                tracker.log_metrics(test_metrics)
                if val_metrics:
                    tracker.log_metrics(val_metrics)
                tracker.set_tags({"model_type": model_name})

                explicit_model_params = _get_explicit_model_params_for_logging(
                    model_name=model_name,
                    model_config=model_config,
                    explicit_param_keys_by_model=explicit_param_keys_by_model,
                )
                if explicit_model_params:
                    tracker.log_params(explicit_model_params)

                # Log grid search metadata if tuning was performed
                grid_search_info = training_results.get("grid_search")
                if grid_search_info:
                    tracker.set_tags({"hyperparameter_tuning": "true"})
                    tracker.log_metrics(
                        {"tuning_best_cv_score": grid_search_info["best_cv_score"]}
                    )
                    tracker.log_params(
                        {
                            "tuning_method": grid_search_info["method"],
                            "tuning_scoring": grid_search_info["scoring"],
                            "tuning_cv_folds": str(grid_search_info["cv_folds"]),
                            "tuning_n_iter": str(grid_search_info["n_iter"]),
                        }
                    )
                    tracker.log_params(
                        {
                            f"tuned__{k.replace('model__', '')}": str(v)
                            for k, v in grid_search_info["best_params"].items()
                        }
                    )

                    # Save cv_results as CSV artifact
                    cv_results_df = grid_search_info["cv_results"]
                    output_dir = Path(paths_config.outputs)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    cv_path = output_dir / "cv_results" / f"{model_name}.csv"
                    cv_path.parent.mkdir(parents=True, exist_ok=True)
                    cv_results_df.to_csv(cv_path, index=False)
                    tracker.log_artifact(str(cv_path), artifact_path="tuning")
                else:
                    tracker.set_tags({"hyperparameter_tuning": "false"})

                tracker.log_model(
                    pipeline,
                    artifact_path=f"nba_{model_name}",
                    input_example=X_train.head(5),
                )

        except Exception as e:
            logger.error(f"Error training {model_name}: {e}", exc_info=True)
            continue

    logger.info(f"\n{'=' * 60}")
    logger.info("Model Comparison - Test Set Results")
    logger.info(f"{'=' * 60}")

    for model_name, model_data in models_to_train.items():
        print_metrics_summary(
            model_data["test_metrics"],
            model_name=model_name,
            split="test",
        )

    if not models_to_train:
        return {}

    # Chosen on validation, not test. Selecting the family by test accuracy made
    # every reported test number optimistic by the size of the max over families.
    # Baselines are a reference line, not candidates: they are fit on the test
    # split itself and have no validation score to compete with.
    candidates = [name for name in models_to_train if "baseline" not in name]
    if not candidates:
        return {}
    best_model_name = max(
        candidates,
        key=lambda name: models_to_train[name]["val_metrics"].get("val_accuracy", 0),
    )
    best_model_data = models_to_train[best_model_name]

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Best Model: {get_model_display_name(best_model_name)}")
    logger.info(
        f"  {format_metrics_line(best_model_data['test_metrics'], prefix='test')}"
    )
    logger.info(f"{'=' * 60}")

    best_metrics = best_model_data["test_metrics"]
    tracker.set_tags({"best_model": best_model_name})
    # The headline test_* metrics are logged after the calibration block below,
    # against whichever pipeline actually ships. Logging them here reported the
    # uncalibrated model's numbers even on runs that shipped a calibrated one.

    # --- Stratified evaluation by season phase ---
    if (
        "games_played_HT" in metadata_test.columns
        and "games_played_VT" in metadata_test.columns
    ):
        y_test_pred_best = best_model_data["pipeline"].predict(X_test)
        phase_metrics = compute_season_phase_metrics(
            y_test, y_test_pred_best, metadata_test
        )
        for _, row in phase_metrics.iterrows():
            phase_label = row["season_phase"].replace("-", "_")
            tracker.log_metrics(
                {f"test_accuracy_phase_{phase_label}": round(row["accuracy"], 4)}
            )
            tracker.log_metrics(
                {f"test_n_games_phase_{phase_label}": int(row["n_games"])}
            )

        record_baseline_data = models_to_train.get("record_baseline")
        if record_baseline_data:
            y_test_pred_baseline = record_baseline_data["pipeline"].predict(
                X_test_baseline.loc[X_test.index]
            )
            baseline_phase_metrics = compute_season_phase_metrics(
                y_test, y_test_pred_baseline, metadata_test
            )
            combined = phase_metrics.merge(
                baseline_phase_metrics[["season_phase", "accuracy"]],
                on="season_phase",
                suffixes=("_model", "_baseline"),
            )
            combined["diff"] = (
                combined["accuracy_model"] - combined["accuracy_baseline"]
            )
            combined = combined.rename(columns={"n_games": "n"})
            logger.info(
                "Season phase accuracy (model vs record baseline):\n"
                + combined[
                    ["season_phase", "n", "accuracy_model", "accuracy_baseline", "diff"]
                ].to_string(index=False)
            )
        else:
            logger.info(
                f"Season phase accuracy:\n{phase_metrics.to_string(index=False)}"
            )
    best_config_params = _get_explicit_model_params_for_logging(
        model_name=best_model_name,
        model_config=model_config,
        explicit_param_keys_by_model=explicit_param_keys_by_model,
    )
    if best_config_params:
        tracker.log_params(best_config_params)

    best_pipeline = best_model_data["pipeline"]

    # --- Probability Calibration ---
    calibrated_pipeline = None
    uncalibrated_proba = None
    best_calibration_method = None
    best_cal_proba = None

    if "baseline" not in best_model_name:
        y_test_proba_uncal = _positive_class_proba(best_pipeline, X_test)
        uncalibrated_proba = y_test_proba_uncal

        brier_uncal = compute_brier_score(y_test, y_test_proba_uncal)
        ece_uncal = compute_ece_ratio(y_test, y_test_proba_uncal)
        tracker.log_metrics(
            {
                "test_brier_score_uncalibrated": round(brier_uncal, 4),
                "test_ece_uncalibrated": round(ece_uncal["ece"], 4),
                "test_ece_null_uncalibrated": round(ece_uncal["ece_null"], 4),
                "test_ece_ratio_uncalibrated": round(ece_uncal["ece_ratio"], 3),
            }
        )
        logger.info(
            f"Uncalibrated — Brier: {brier_uncal:.4f}, "
            f"ECE: {ece_uncal['ece']:.4f} "
            f"({ece_uncal['ece_ratio']:.2f}x the perfectly-calibrated null "
            f"{ece_uncal['ece_null']:.4f})"
        )

        # Which calibrator to use is *chosen* on validation and only *reported*
        # on test. Choosing it by test Brier made the shipped model a function
        # of the test set. The probe runs out-of-fold across all of validation:
        # a prefit calibrator scored on the rows it was fit to would always
        # favour isotonic, which can memorise them.
        calibration_results = {}
        brier_uncal_val = None
        if len(X_val) < MIN_CALIBRATION_PROBE_ROWS:
            logger.warning(
                f"Validation split has {len(X_val)} rows; need at least "
                f"{MIN_CALIBRATION_PROBE_ROWS} to choose a calibrator without "
                "touching test. Keeping the uncalibrated model."
            )
        else:
            oof_uncal = _oof_calibrated_proba(best_pipeline, X_val, y_val, None)
            brier_uncal_val = compute_brier_score(y_val, oof_uncal)
            logger.info(
                f"Uncalibrated — val Brier (out-of-fold, n={len(X_val)}): "
                f"{brier_uncal_val:.4f}"
            )

            for method in ["sigmoid", "isotonic"]:
                try:
                    oof_cal = _oof_calibrated_proba(
                        best_pipeline, X_val, y_val, method
                    )
                    brier_val = compute_brier_score(y_val, oof_cal)
                    val_delta, val_se = _paired_brier_margin(
                        y_val, oof_cal, oof_uncal
                    )

                    # The probe only scored the method. The model that ships is
                    # refit on the whole validation split.
                    cal_model = CalibratedClassifierCV(
                        best_pipeline, method=method, cv="prefit"
                    )
                    cal_model.fit(X_val, y_val)
                    y_test_proba_cal = _positive_class_proba(cal_model, X_test)
                    brier_cal = compute_brier_score(y_test, y_test_proba_cal)
                    ece_cal = compute_ece_ratio(y_test, y_test_proba_cal)
                    calibration_results[method] = {
                        "model": cal_model,
                        "val_brier": brier_val,
                        "val_delta": val_delta,
                        "val_se": val_se,
                        "brier": brier_cal,
                        "ece": ece_cal["ece"],
                        "proba": y_test_proba_cal,
                    }
                    tracker.log_metrics(
                        {
                            f"val_brier_score_{method}": round(brier_val, 4),
                            f"val_brier_delta_{method}": round(val_delta, 5),
                            f"val_brier_delta_se_{method}": round(val_se, 5),
                            f"test_brier_score_{method}": round(brier_cal, 4),
                            f"test_ece_{method}": round(ece_cal["ece"], 4),
                            f"test_ece_null_{method}": round(ece_cal["ece_null"], 4),
                            f"test_ece_ratio_{method}": round(ece_cal["ece_ratio"], 3),
                        }
                    )
                    logger.info(
                        f"Calibrated [{method}] — val Brier: {brier_val:.4f} "
                        f"({val_delta:+.5f} vs uncalibrated, SE {val_se:.5f}), "
                        f"test Brier: {brier_cal:.4f}, "
                        f"test ECE: {ece_cal['ece']:.4f} "
                        f"({ece_cal['ece_ratio']:.2f}x null)"
                    )
                except Exception as e:
                    logger.warning(f"Calibration with {method} failed: {e}")

        if calibration_results:
            best_cal_method = min(
                calibration_results, key=lambda m: calibration_results[m]["val_brier"]
            )
            best_calibration_method = best_cal_method
            best_cal_proba = calibration_results[best_cal_method]["proba"]
            best_delta = calibration_results[best_cal_method]["val_delta"]
            best_se = calibration_results[best_cal_method]["val_se"]
            # Adopting on *any* improvement shipped a calibrator on a 0.0003 Brier
            # win — noise — and that coin flip pushed test ECE from 0.96x to 1.43x
            # its perfectly-calibrated null. The uncalibrated model is already well
            # calibrated, so it keeps the tie: a calibrator has to beat it by more
            # than one standard error of the paired difference to displace it.
            if brier_uncal_val is not None and best_delta < -best_se:
                calibrated_pipeline = calibration_results[best_cal_method]["model"]
                # Propagate feature_names_in_ for prediction alignment
                if hasattr(best_pipeline, "feature_names_in_"):
                    calibrated_pipeline.feature_names_in_ = (
                        best_pipeline.feature_names_in_
                    )
                best_pipeline = calibrated_pipeline
                best_model_data["pipeline"] = best_pipeline
                tracker.set_tags(
                    {
                        "calibration_method": best_calibration_method,
                        "calibration_improved": True,
                    }
                )
                logger.info(
                    f"Using calibrated model ({best_calibration_method}): "
                    f"val Brier {best_delta:+.5f} beats uncalibrated by more than "
                    f"its SE ({best_se:.5f})"
                )
            else:
                tracker.set_tags({"calibration_improved": False})
                logger.info(
                    f"Best calibrator ({best_cal_method}) moved validation Brier by "
                    f"{best_delta:+.5f} against an SE of {best_se:.5f} — inside the "
                    "noise. Keeping the uncalibrated model."
                )

    # Headline test_* metrics, computed against the pipeline that actually ships.
    # best_model_data["test_metrics"] was measured before calibration, so on a run
    # that adopted a calibrator it described a model that never left the building.
    if calibrated_pipeline is not None:
        y_test_proba_best = _positive_class_proba(best_pipeline, X_test)
        best_metrics = {
            f"test_{k}": v
            for k, v in compute_classification_metrics(
                y_test, best_pipeline.predict(X_test), y_pred_proba=y_test_proba_best
            ).items()
        }
        ece_best = compute_ece_ratio(y_test, y_test_proba_best)
        best_metrics["test_ece_null"] = round(ece_best["ece_null"], 4)
        best_metrics["test_ece_ratio"] = round(ece_best["ece_ratio"], 3)
    elif uncalibrated_proba is not None:
        best_metrics = dict(best_metrics)
        best_metrics["test_ece_null"] = round(ece_uncal["ece_null"], 4)
        best_metrics["test_ece_ratio"] = round(ece_uncal["ece_ratio"], 3)
    tracker.log_metrics(best_metrics)
    logger.info(f"Shipped model — {format_metrics_line(best_metrics, prefix='test')}")

    if eval_config.save_visualizations:
        logger.info("Generating visualizations...")
        output_dir = Path(paths_config.outputs)
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        y_test_pred = best_pipeline.predict(X_test)

        class_names = sorted(y.unique())
        fig_cm = plot_confusion_matrix(
            y_test,
            y_test_pred,
            class_names=class_names,
            title=f"Test Set Confusion Matrix - {best_model_name}",
        )
        fig_cm.savefig(viz_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close(fig_cm)

        y_test_proba = None
        if "baseline" not in best_model_name:
            if hasattr(best_pipeline, "named_steps"):
                trained_model = best_pipeline.named_steps["model"]
                preprocessor_fitted = best_pipeline.named_steps["preprocessor"]
            elif hasattr(best_pipeline, "estimator") and hasattr(
                best_pipeline.estimator, "named_steps"
            ):
                trained_model = best_pipeline.estimator.named_steps["model"]
                preprocessor_fitted = best_pipeline.estimator.named_steps[
                    "preprocessor"
                ]
            else:
                trained_model = best_pipeline
                preprocessor_fitted = None

            # Resolved for both the importance and SHAP plots below.
            if preprocessor_fitted is not None and hasattr(
                preprocessor_fitted, "get_feature_names_out"
            ):
                try:
                    feature_names = clean_feature_names(
                        list(
                            preprocessor_fitted.get_feature_names_out(
                                list(X_train.columns)
                            )
                        )
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not extract feature names from preprocessor: {e}. Using original."
                    )
                    feature_names = (
                        numerical_features
                        + (boolean_features or [])
                        + (categorical_features or [])
                    )
            else:
                feature_names = (
                    numerical_features
                    + (boolean_features or [])
                    + (categorical_features or [])
                )

            if hasattr(trained_model, "feature_importances_"):
                if len(feature_names) != len(trained_model.feature_importances_):
                    logger.error(
                        f"Feature names length ({len(feature_names)}) does not match "
                        f"feature importances length ({len(trained_model.feature_importances_)}). "
                        f"Skipping feature importance plot."
                    )
                else:
                    fig_importance = plot_feature_importance(
                        trained_model,
                        feature_names,
                        top_n=20,
                        title=f"Feature Importance - {best_model_name}",
                    )
                    fig_importance.savefig(
                        viz_dir / "feature_importance.png", dpi=300, bbox_inches="tight"
                    )
                    plt.close(fig_importance)

            # --- SHAP summary plot ---
            try:
                from src.ml.evaluation.visualization import plot_shap_summary

                X_test_transformed = preprocessor_fitted.transform(X_test)
                if hasattr(X_test_transformed, "toarray"):
                    X_test_transformed = X_test_transformed.toarray()
                elif hasattr(X_test_transformed, "values"):
                    X_test_transformed = X_test_transformed.values
                fig_shap = plot_shap_summary(
                    trained_model,
                    X_test_transformed,
                    feature_names,
                    title=f"SHAP Feature Contribution - {best_model_name}",
                    top_n=20,
                )
                fig_shap.savefig(
                    viz_dir / "shap_summary.png", dpi=300, bbox_inches="tight"
                )
                plt.close(fig_shap)
                logger.info("SHAP summary plot saved.")
            except Exception as e:
                logger.warning(f"Could not generate SHAP plot: {e}")

            y_test_proba = best_pipeline.predict_proba(X_test)
            if y_test_proba.ndim > 1:
                y_test_proba = y_test_proba[:, 1]
            fig_roc = plot_roc_curve(
                y_test,
                y_test_proba,
                title=f"ROC Curve - {best_model_name}",
            )
            fig_roc.savefig(viz_dir / "roc_curve.png", dpi=300, bbox_inches="tight")
            plt.close(fig_roc)

            fig_acc_bin = plot_prediction_accuracy_by_bin(
                y_test,
                y_test_proba,
                title=f"Prediction Accuracy by Probability Bin - {best_model_name}",
            )
            fig_acc_bin.savefig(
                viz_dir / "prediction_accuracy_by_bin.png", dpi=300, bbox_inches="tight"
            )
            plt.close(fig_acc_bin)

            # Calibration curve
            if uncalibrated_proba is not None and best_cal_proba is not None:
                fig_cal = plot_calibration_curve(
                    y_test,
                    uncalibrated_proba,
                    best_cal_proba,
                    calibration_method=best_calibration_method,
                    title=f"Calibration Curve - {best_model_name}",
                )
                fig_cal.savefig(
                    viz_dir / "calibration_curve.png", dpi=300, bbox_inches="tight"
                )
                plt.close(fig_cal)
                logger.info("Calibration curve saved.")

        logger.info(f"Visualizations saved to {viz_dir}")

        # Metadata-based analysis: error patterns, accuracy breakdowns, calibration
        logger.info("Generating metadata-based analysis...")
        _record_baseline_data = models_to_train.get("record_baseline")
        _y_pred_baseline = (
            _record_baseline_data["pipeline"].predict(X_test_baseline)
            if _record_baseline_data
            else None
        )
        generate_analysis(
            y_true=y_test,
            y_pred=y_test_pred,
            y_pred_proba=y_test_proba,
            metadata=metadata_test,
            output_dir=output_dir,
            model_name=best_model_name,
            y_pred_baseline=_y_pred_baseline,
        )
        logger.info(
            f"Analysis saved to {output_dir / 'analysis'} and {output_dir / 'tables'}"
        )

        tracker.log_artifacts(str(output_dir), artifact_path="outputs")

    if paths_config.save_local_models:
        logger.info(f"Saving best model ({best_model_name}) locally...")
        registry = ModelRegistry(Path(paths_config.model_registry))
        model_path = registry.save(
            model=best_pipeline,
            model_name=f"nba_classification_{best_model_name}",
            task_type="classification",
            metrics=best_model_data["test_metrics"],
            feature_names=list(X_train.columns),
        )
        logger.info(f"Best model saved to: {model_path}")
    else:
        logger.info("Skipping local model save; MLflow handles model logging.")

    should_register = config.mlflow.register_model
    if should_register:
        registered_model_name = (
            f"nba_classification_{best_model_name}"
        )
        logger.info(
            f"Registering model '{registered_model_name}' in MLflow Model Registry"
        )
    else:
        registered_model_name = None
        logger.info(
            "Model will be logged but not registered. Use run-based URI for access."
        )

    artifact_path = f"nba_{best_model_name}"
    tracker.log_model(
        best_pipeline,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name,
        input_example=X_train.head(5),
        dataset_name=splits.date_range_name,
    )

    run_id = tracker.get_run_id()
    run_model_uri = None
    if run_id:
        run_model_uri = f"runs:/{run_id}/{artifact_path}"
        logger.info(f"Model logged at run URI: {run_model_uri}")
        tags = {"model_run_uri": run_model_uri, "model_name": best_model_name}
        if registered_model_name:
            tags["registered_model_name"] = registered_model_name
        tracker.set_tags(tags)

    return {
        "best_model_name": best_model_name,
        "best_model_data": best_model_data,
        "models_to_train": models_to_train,
        "registered_model_name": registered_model_name,
        "run_model_uri": run_model_uri,
    }
