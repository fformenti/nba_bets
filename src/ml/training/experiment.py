from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config.paths import CONFIGS_TRAIN_DIR, REGULAR_SEASON_GAMES_FEATURES_PATH
from src.ml.config.loader import load_yaml_config
from src.ml.config.schema import ExperimentConfig
from src.ml.datasets.splitters import temporal_split
from sklearn.calibration import CalibratedClassifierCV

from src.ml.evaluation.metrics import (
    compute_brier_score,
    compute_ece,
    format_metrics_line,
    get_conference_display_name,
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
from src.ml.features.engineering import (
    apply_conference_features,
    create_delta_features,
    identify_feature_types,
    resolve_feature_columns,
)
from src.ml.features.preprocessing import create_preprocessing_pipeline
from src.ml.models.baseline import RecordDifferenceBaseline
from src.ml.models.registry import ModelRegistry
from src.utils.logging_config import get_logger

from .data_prep import filter_minimum_games_played, load_and_validate_data, prepare_data
from .model_factory import clean_feature_names
from .runners import train_baseline_model, train_model_with_config

logger = get_logger(__name__)


def _extract_explicit_model_param_keys(
    config_path: Optional[Path],
) -> dict[str, set[str]]:
    """Read `_defaults.yaml` and return explicit top-level param keys per model."""
    defaults_path = (
        (config_path.parent / "_defaults.yaml")
        if config_path is not None
        else (CONFIGS_TRAIN_DIR / "_defaults.yaml")
    )
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
    explicit_param_keys_by_model = _extract_explicit_model_param_keys(config_path)
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
    date_column = data_config.date_column

    metadata_columns = feat_eng_config.metadata_columns
    intermediate_columns = feat_eng_config.intermediate_columns

    minimum_games_train = filters_config.minimum_games_train
    minimum_games_test = filters_config.minimum_games_test
    conference_filter = filters_config.conference_filter

    test_size = split_config.test_size
    val_size = split_config.val_size
    random_state = split_config.random_state

    if conference_filter not in ["same", "different", "all"]:
        raise ValueError(
            f"conference_filter must be 'same', 'different', or 'all', got '{conference_filter}'"
        )

    min_season = filters_config.min_season

    tracker.log_params(
        {
            "target_column": target_column,
            "split_method": split_config.method,
            "test_size": test_size,
            "val_size": val_size,
            "minimum_games_train": minimum_games_train,
            "minimum_games_test": minimum_games_test,
            "conference_filter": conference_filter,
            "min_season": min_season,
            "scaling_method": config.preprocessing.scaling_method,
            "sample_weighting_enabled": weighting_config.enabled,
            "sample_weighting_K": weighting_config.saturation_K,
            "season_decay_enabled": weighting_config.season_decay_enabled,
            "season_decay_lambda": weighting_config.season_decay_lambda,
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
            "conference_filter": conference_filter,
        }
    )
    if config.tags:
        tracker.set_tags(config.tags)
    if config.description:
        tracker.set_tags({"mlflow.note.content": config.description})

    games_enriched = load_and_validate_data(
        data_path=data_path, target_column=target_column, date_column=date_column
    )

    if min_season is not None:
        if "season" not in games_enriched.columns:
            raise ValueError(
                "DataFrame must contain 'season' column for min_season filtering"
            )
        games_enriched = games_enriched[games_enriched["season"] >= min_season].copy()
        logger.info(
            f"Filtered to {len(games_enriched)} games (min_season: {min_season})"
        )

    if conference_filter == "different":
        games_enriched = games_enriched[
            games_enriched["hometeamConference"] != games_enriched["awayteamConference"]
        ].copy()
        logger.info(f"Filtered to {len(games_enriched)} games (different conferences)")
    elif conference_filter == "same":
        games_enriched = games_enriched[
            games_enriched["hometeamConference"] == games_enriched["awayteamConference"]
        ].copy()
        logger.info(f"Filtered to {len(games_enriched)} games (same conference)")
    else:
        logger.info(f"Using all {len(games_enriched)} games (no conference filter)")

    games_enriched = filter_minimum_games_played(
        cast(pd.DataFrame, games_enriched), minimum_games_test
    )

    date_range_name = None
    if date_column and date_column in games_enriched.columns:
        dates = pd.to_datetime(games_enriched[date_column], errors="coerce").dropna()
        if not dates.empty:
            date_range_name = f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
            tracker.log_dataset(
                df=games_enriched,
                source=str(data_path),
                name=date_range_name,
                targets=target_column,
                context="training",
            )

    df, y, metadata = prepare_data(
        df=games_enriched,
        target_column=target_column,
        drop_na=data_config.drop_na,
        metadata_columns=metadata_columns,
    )

    # Special Dataframe for baseline models
    _, _, X_test_baseline, _, _, y_test_baseline = temporal_split(
        df, y, date_column=date_column, test_size=test_size, val_size=val_size
    )
    X_test_baseline["record_L82_delta"] = (
        X_test_baseline["record_L82_HT"] - X_test_baseline["record_L82_VT"]
    )
    X_test_baseline["pts_diff_avg_L82_delta"] = (
        X_test_baseline["pts_diff_avg_L82_HT"] - X_test_baseline["pts_diff_avg_L82_VT"]
    )
    # X_test_baseline = X_test_baseline[["record_L82_HT", "record_L82_VT", "pts_diff_avg_L82_HT", "pts_diff_avg_L82_VT"]]

    df = create_delta_features(df, feat_eng_config.features)
    df = apply_conference_features(df, conference_filter)
    # if config.feature_engineering.momentum_pairs:
    #     df = create_momentum_features(df, config.feature_engineering.momentum_pairs)

    if feat_eng_config.selection_mode == "inclusion":
        feature_columns = resolve_feature_columns(
            feat_eng_config.features,
            conference_filter,
            momentum_pairs=config.feature_engineering.momentum_pairs or None,
        )
        keep_cols = feature_columns + [date_column]
        available = [c for c in keep_cols if c in df.columns]
        missing_features = [c for c in feature_columns if c not in df.columns]
        if missing_features:
            formatted_missing = "\n".join(
                f"  - MISSING FEATURE: {feature}" for feature in missing_features
            )
            logger.warning(
                "\n"
                + "!" * 90
                + "\n"
                + "!!! CRITICAL WARNING: EXPECTED FEATURE COLUMNS WERE NOT FOUND IN DATA !!!\n"
                + "This will likely degrade model quality and should be investigated immediately.\n"
                + formatted_missing
                + "\n"
                + "!" * 90
            )
        X = df[available]
    else:
        exclude_cols = metadata_columns + intermediate_columns + [target_column]
        cols_to_drop = [col for col in exclude_cols if col in df.columns]
        X = df.drop(columns=cols_to_drop)

    if date_column not in X.columns:
        raise ValueError(f"Date column '{date_column}' required for temporal split")
    X_train, X_val, X_test, y_train, y_val, y_test = temporal_split(
        X, y, date_column=date_column, test_size=test_size, val_size=val_size
    )
    X_train = cast(pd.DataFrame, X_train.drop(columns=[date_column]))
    X_val = X_val.drop(columns=[date_column])
    X_test = X_test.drop(columns=[date_column])

    if minimum_games_train > minimum_games_test:
        train_gp = games_enriched.loc[X_train.index]
        train_mask = (train_gp["games_played_HT"] > minimum_games_train) & (
            train_gp["games_played_VT"] > minimum_games_train
        )
        X_train = X_train.loc[train_mask]
        y_train = cast(pd.Series, y_train.loc[train_mask])

        val_gp = games_enriched.loc[X_val.index]
        val_mask = (val_gp["games_played_HT"] > minimum_games_train) & (
            val_gp["games_played_VT"] > minimum_games_train
        )
        X_val = cast(pd.DataFrame, X_val.loc[val_mask])
        y_val = cast(pd.Series, y_val.loc[val_mask])

    metadata_test = metadata.loc[X_test.index].copy()

    # --- Sample weight computation ---
    train_sample_weight = None
    if weighting_config.enabled:
        K = weighting_config.saturation_K
        train_gp_ht = cast(
            pd.Series,
            pd.to_numeric(
                metadata.loc[X_train.index, "games_played_HT"], errors="coerce"
            ),
        )
        train_gp_vt = cast(
            pd.Series,
            pd.to_numeric(
                metadata.loc[X_train.index, "games_played_VT"], errors="coerce"
            ),
        )
        min_gp_train = np.minimum(
            train_gp_ht.to_numpy(dtype=np.float64, copy=False),
            train_gp_vt.to_numpy(dtype=np.float64, copy=False),
        )
        train_sample_weight = np.clip(min_gp_train / float(K), 0.0, 1.0)
        logger.info(
            f"Within-season weighting (K={K}): "
            f"min={train_sample_weight.min():.2f}, "
            f"mean={train_sample_weight.mean():.2f}, "
            f"pct_full={(train_sample_weight == 1.0).mean() * 100:.1f}%"
        )

    if weighting_config.season_decay_enabled:
        lam = weighting_config.season_decay_lambda
        train_seasons = metadata.loc[X_train.index, "season"]
        season_order = sorted(train_seasons.unique())
        season_to_idx = {s: i for i, s in enumerate(season_order)}
        max_idx = len(season_order) - 1
        season_indices = train_seasons.map(season_to_idx).to_numpy(dtype=np.float64)
        cross_season_weight = np.exp(-lam * (max_idx - season_indices))
        if train_sample_weight is None:
            train_sample_weight = cross_season_weight
        else:
            train_sample_weight = train_sample_weight * cross_season_weight
        logger.info(
            f"Cross-season decay (lambda={lam}): "
            f"min={cross_season_weight.min():.3f}, "
            f"mean={cross_season_weight.mean():.3f}, "
            f"oldest={season_order[0]}, newest={season_order[-1]}"
        )

    logger.info(
        f"Features BEFORE selection ({len(X_train.columns)}): {sorted(X_train.columns.tolist())}"
    )

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
                top_n=30,
                title=f"Boruta-SHAP Feature Selection ({conference_filter})",
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
    #! TODO: Make this feature available even when it's not passed to train config
    # logger.info(f"\n{'=' * 60}")
    # logger.info("Point Differential Baseline")
    # logger.info(f"{'=' * 60}")

    # baseline_model = PointDifferentialBaseline(feature_column="pts_diff_avg_L82_delta")

    # models_to_train[baseline_model.name] = {
    #     "pipeline": baseline_model,
    #     "trainer": baseline_trainer,
    #     "training_results": {},
    #     "test_metrics": baseline_test_metrics,
    # }

    # with tracker.child_run(baseline_model.name):
    #     tracker.log_metrics(baseline_test_metrics)
    #     tracker.set_tags({"model_type": "baseline"})

    model_names = list(models_to_train.keys()) + list(config.model.train_models)
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
            models_to_train[model_name] = {
                "pipeline": pipeline,
                "trainer": trainer,
                "training_results": training_results,
                "test_metrics": test_metrics,
            }

            with tracker.child_run(model_name):
                tracker.log_metrics(test_metrics)
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
                    artifact_path=f"nba_{model_name}_{conference_filter}",
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
            conference_filter=conference_filter,
        )

    if not models_to_train:
        return {}

    best_model_name = max(
        models_to_train.keys(),
        key=lambda name: models_to_train[name]["test_metrics"].get("test_accuracy", 0),
    )
    best_model_data = models_to_train[best_model_name]

    conf_label = get_conference_display_name(conference_filter)
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Best Model: {get_model_display_name(best_model_name)} ({conf_label})")
    logger.info(
        f"  {format_metrics_line(best_model_data['test_metrics'], prefix='test')}"
    )
    logger.info(f"{'=' * 60}")

    best_metrics = best_model_data["test_metrics"]
    tracker.set_tags({"best_model": best_model_name})
    tracker.log_metrics(best_metrics)

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
                X_test_baseline
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

    if "baseline" not in best_model_name:
        y_test_proba_uncal = best_pipeline.predict_proba(X_test)
        if y_test_proba_uncal.ndim > 1:
            y_test_proba_uncal = y_test_proba_uncal[:, 1]
        uncalibrated_proba = y_test_proba_uncal

        brier_uncal = compute_brier_score(y_test, y_test_proba_uncal)
        ece_uncal = compute_ece(y_test, y_test_proba_uncal)
        tracker.log_metrics(
            {
                "test_brier_score_uncalibrated": round(brier_uncal, 4),
                "test_ece_uncalibrated": round(ece_uncal, 4),
            }
        )
        logger.info(f"Uncalibrated — Brier: {brier_uncal:.4f}, ECE: {ece_uncal:.4f}")

        calibration_results = {}
        for method in ["sigmoid", "isotonic"]:
            try:
                cal_model = CalibratedClassifierCV(
                    best_pipeline, method=method, cv="prefit"
                )
                cal_model.fit(X_val, y_val)
                y_test_proba_cal = cal_model.predict_proba(X_test)
                if y_test_proba_cal.ndim > 1:
                    y_test_proba_cal = y_test_proba_cal[:, 1]
                brier_cal = compute_brier_score(y_test, y_test_proba_cal)
                ece_cal = compute_ece(y_test, y_test_proba_cal)
                calibration_results[method] = {
                    "model": cal_model,
                    "brier": brier_cal,
                    "ece": ece_cal,
                    "proba": y_test_proba_cal,
                }
                tracker.log_metrics(
                    {
                        f"test_brier_score_{method}": round(brier_cal, 4),
                        f"test_ece_{method}": round(ece_cal, 4),
                    }
                )
                logger.info(
                    f"Calibrated [{method}] — Brier: {brier_cal:.4f}, ECE: {ece_cal:.4f}"
                )
            except Exception as e:
                logger.warning(f"Calibration with {method} failed: {e}")

        if calibration_results:
            best_cal_method = min(
                calibration_results, key=lambda m: calibration_results[m]["brier"]
            )
            if calibration_results[best_cal_method]["brier"] < brier_uncal:
                best_calibration_method = best_cal_method
                calibrated_pipeline = calibration_results[best_cal_method]["model"]
                # Propagate feature_names_in_ for prediction alignment
                if hasattr(best_pipeline, "feature_names_in_"):
                    calibrated_pipeline.feature_names_in_ = (
                        best_pipeline.feature_names_in_
                    )
                best_pipeline = calibrated_pipeline
                best_model_data["pipeline"] = best_pipeline
                tracker.set_tags({"calibration_method": best_calibration_method})
                logger.info(f"Using calibrated model ({best_calibration_method})")
            else:
                logger.info(
                    "Calibration did not improve Brier score; keeping uncalibrated model."
                )

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
            title=f"Test Set Confusion Matrix - {best_model_name} ({conference_filter})",
        )
        fig_cm.savefig(viz_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")

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

            if hasattr(trained_model, "feature_importances_"):
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
                        title=f"Feature Importance - {best_model_name} ({conference_filter})",
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
                    title=f"SHAP Feature Contribution - {best_model_name} ({conference_filter})",
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
                title=f"ROC Curve - {best_model_name} ({conference_filter})",
            )
            fig_roc.savefig(viz_dir / "roc_curve.png", dpi=300, bbox_inches="tight")

            fig_acc_bin = plot_prediction_accuracy_by_bin(
                y_test,
                y_test_proba,
                title=f"Prediction Accuracy by Probability Bin - {best_model_name} ({conference_filter})",
            )
            fig_acc_bin.savefig(
                viz_dir / "prediction_accuracy_by_bin.png", dpi=300, bbox_inches="tight"
            )
            plt.close(fig_acc_bin)

            # Calibration curve
            if calibrated_pipeline is not None and uncalibrated_proba is not None:
                calibrated_proba = best_pipeline.predict_proba(X_test)
                if calibrated_proba.ndim > 1:
                    calibrated_proba = calibrated_proba[:, 1]
                fig_cal = plot_calibration_curve(
                    y_test,
                    uncalibrated_proba,
                    calibrated_proba,
                    calibration_method=best_calibration_method,
                    title=f"Calibration Curve - {best_model_name} ({conference_filter})",
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
            conference_filter=conference_filter,
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
            model_name=f"nba_classification_{best_model_name}_{conference_filter}",
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
            f"nba_classification_{best_model_name}_{conference_filter}"
        )
        logger.info(
            f"Registering model '{registered_model_name}' in MLflow Model Registry"
        )
    else:
        registered_model_name = None
        logger.info(
            "Model will be logged but not registered. Use run-based URI for access."
        )

    artifact_path = f"nba_{best_model_name}_{conference_filter}"
    tracker.log_model(
        best_pipeline,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name,
        input_example=X_train.head(5),
        dataset_name=date_range_name,
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
        "conference_filter": conference_filter,
        "best_model_name": best_model_name,
        "best_model_data": best_model_data,
        "models_to_train": models_to_train,
        "registered_model_name": registered_model_name,
        "run_model_uri": run_model_uri,
    }
