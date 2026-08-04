"""Predict upcoming games with the trained per-conference classifiers.

Each conference matchup type ('same', 'different', 'all') has its own model, so
a slate is routed three ways and the parts concatenated. The experiment config
that trained each model is read back from its MLflow run, guaranteeing the
inference-time feature set matches the training-time one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import mlflow
import numpy as np
import pandas as pd

from src.config.paths import DEFAULT_PREDICT_CONFIG_PATH, PROJECT_ROOT
from src.ml.config.loader import load_experiment_config, load_prediction_config
from src.ml.config.schema import ExperimentConfig
from src.ml.prediction.features import (
    build_features_for_prediction,
    fix_upcoming_games_cols,
    prepare_features_for_model,
)
from src.ml.prediction.io import load_upcoming_games
from src.ml.tracking.mlflow_tracker import (
    MLflowTracker,
    load_experiment_config_from_model_uri,
)
from src.ml.training.data_prep import load_and_validate_data
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

# Predictions accumulate across runs, so a game must be uniquely identified.
# Re-predicting the same game replaces the earlier row rather than appending a
# duplicate — the accuracy scorecard in src/monitoring counts these rows.
PREDICTION_KEY = ["gameId", "conference_filter"]


def upsert_predictions(new_predictions: pd.DataFrame, output_path: Path) -> int:
    """Merge predictions into the running CSV, replacing any earlier row per game.

    The previous behaviour appended unconditionally, so re-running a slate (a
    routine thing — the model is retrained, or a run is repeated) silently
    duplicated every game. Anything counting rows downstream, in particular the
    accuracy scorecard, would then double-count those games.

    Returns the total row count after the merge.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined = new_predictions
    if output_path.exists():
        existing = pd.read_csv(output_path)
        combined = pd.concat([existing, new_predictions], ignore_index=True)

    key = [col for col in PREDICTION_KEY if col in combined.columns]
    if key:
        before = len(combined)
        combined = combined.drop_duplicates(subset=key, keep="last")
        if before != len(combined):
            logger.info(f"Replaced {before - len(combined)} superseded prediction(s)")
    else:
        logger.warning(f"None of {PREDICTION_KEY} present; writing without de-duplication.")

    sort_column = next(
        (col for col in ("gameDate", "gameDateOnlyStr") if col in combined.columns), None
    )
    if sort_column:
        combined = combined.sort_values(sort_column)

    combined.to_csv(output_path, index=False)
    return len(combined)


def _collect_metadata(
    upcoming_games: pd.DataFrame,
    upcoming_with_features: pd.DataFrame,
    metadata_columns: list[str],
) -> pd.DataFrame:
    """Gather the config's metadata columns for the predicted rows.

    The experiment's ``metadata_columns`` describe a *played* game, so some of
    them (final scores, locations, games-played counts) only exist after feature
    merging, and a few never exist for a game that has not happened. Take each
    column from whichever frame has it and skip the rest, rather than failing —
    metadata is for identifying the prediction, not for making it.
    """
    index = upcoming_with_features.index
    columns = {}
    missing = []

    for column in metadata_columns:
        if column in upcoming_games.columns:
            columns[column] = upcoming_games.loc[index, column]
        elif column in upcoming_with_features.columns:
            columns[column] = upcoming_with_features.loc[index, column]
        else:
            missing.append(column)

    if missing:
        logger.info(f"Metadata columns unavailable for upcoming games: {missing}")

    return pd.DataFrame(columns, index=index)


def _load_model(
    model_uri: str,
    tracking_uri: Optional[str] = None,
) -> Tuple[object, Optional[list[str]]]:
    """Load model from MLflow."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model = mlflow.sklearn.load_model(model_uri)
    feature_names = getattr(model, "feature_names_in_", None)
    return model, list(feature_names) if feature_names is not None else None


def _align_features(
    features: pd.DataFrame,
    feature_names: Optional[list[str]],
    allow_missing: bool,
) -> pd.DataFrame:
    if feature_names is None:
        return features

    missing = [col for col in feature_names if col not in features.columns]
    if missing and not allow_missing:
        raise ValueError(f"Missing required features: {missing}")

    # Log missing features
    if missing:
        logger.info(f"Missing features that will be filled with NaN: {missing}")

    aligned = features.copy()
    for col in missing:
        aligned[col] = np.nan  # Use numpy NaN instead of pd.NA for sklearn compatibility

    # Reorder columns to match feature_names
    aligned = aligned[feature_names]

    # Check for missing values and log details
    missing_mask = aligned.isna().any(axis=1)
    if missing_mask.any():
        n_rows_with_missing = missing_mask.sum()
        logger.info(f"Rows with missing values: {n_rows_with_missing} out of {len(aligned)}")

        # Log which columns have missing values
        missing_per_column = aligned.isna().sum()
        columns_with_missing = missing_per_column[missing_per_column > 0]
        if len(columns_with_missing) > 0:
            logger.info(f"Features with missing values:\n{columns_with_missing.to_dict()}")

        # Log row indices with missing values
        rows_with_missing = aligned[missing_mask].index.tolist()
        logger.debug(f"Row indices with missing values: {rows_with_missing}")
    else:
        logger.info("No missing values found in aligned features")

    return aligned


def _predict(model: object, features: pd.DataFrame) -> Tuple[pd.Series, Optional[pd.Series]]:
    # Convert pandas NA values to numpy NaN for sklearn compatibility
    # This handles the case where missing features were set to pd.NA
    # fillna converts pd.NA to np.nan for sklearn compatibility
    features = features.fillna(np.nan)
    predictions = model.predict(features)
    preds = pd.Series(predictions, index=features.index)

    proba = None
    if hasattr(model, "predict_proba"):
        proba_raw = model.predict_proba(features)
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
            if 1 in classes:
                class_index = classes.index(1)
                proba = pd.Series(proba_raw[:, class_index], index=features.index)
            else:
                proba = pd.Series(proba_raw.max(axis=1), index=features.index)
        else:
            proba = pd.Series(proba_raw.max(axis=1), index=features.index)

    return preds, proba


def run_prediction_pipeline(
    config_path: Optional[Path] = None,
):
    """
    Main prediction function.

    Parameters
    ----------
    config_path : Path, optional
        Path to prediction configuration YAML file. If None, uses default configuration.
    """
    # Setup logging
    setup_logging(level="INFO")

    # Load prediction configuration
    if config_path is None:
        config_path = DEFAULT_PREDICT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(f"Prediction config file not found: {config_path}")

    prediction_config = load_prediction_config(config_path)
    logger.info(f"Loaded prediction configuration from {config_path}")

    mlflow_config = prediction_config.mlflow

    # Start MLflow tracking
    with MLflowTracker(
        experiment_name=mlflow_config.experiment_name,
        run_name=config_path.stem if config_path else None,
        tracking_uri=mlflow_config.tracking_uri,
        log_model=False,
    ) as tracker:
        # Log configuration
        tracker.log_params(
            {
                "model_uris": str(prediction_config.model_uris),
                "input_dir": prediction_config.paths.input_dir,
                "output_path": prediction_config.paths.output,
                "features_path": prediction_config.paths.features,
                "allow_missing_features": prediction_config.allow_missing_features,
            }
        )

        # Data loading
        data_config = prediction_config.data
        target_column = data_config.target_column
        date_column = data_config.date_column

        # Load upcoming games
        input_dir = PROJECT_ROOT / prediction_config.paths.input_dir
        logger.info(f"Loading upcoming games from {input_dir}")
        upcoming_games = load_upcoming_games(
            input_dir,
            max_files=prediction_config.max_files,
        )

        if upcoming_games.empty:
            logger.warning("No upcoming games found to predict")
            return

        logger.info(f"Loaded {len(upcoming_games)} upcoming games")

        # Prepare upcoming games
        upcoming_games = fix_upcoming_games_cols(upcoming_games)

        # Load historical features
        features_path = PROJECT_ROOT / prediction_config.paths.features
        logger.info(f"Loading historical features from {features_path}")

        historical_features = load_and_validate_data(
            data_path=features_path,
            target_column=target_column,
            date_column=date_column,
        )

        # Filter historical features to relevant seasons
        upcoming_seasons = upcoming_games["season"].dropna().unique().tolist()
        if upcoming_seasons:
            historical_features = historical_features[
                historical_features["season"].isin(upcoming_seasons)
            ].copy()
            logger.info(f"Filtered historical features to seasons: {upcoming_seasons}")

        # Route each game to the appropriate model
        results_parts = []
        for conference_filter in ["same", "different", "all"]:
            if conference_filter not in prediction_config.model_uris:
                logger.warning(f"No model URI for '{conference_filter}', skipping")
                continue

            model_uri = prediction_config.model_uris[conference_filter]

            # Load the experiment config that was used to train this model.
            # Primary: from the MLflow run artifact (guaranteed to match the model).
            # Fallback: from a local file via feature_config_path.
            try:
                config_dict = load_experiment_config_from_model_uri(
                    model_uri=model_uri,
                    tracking_uri=mlflow_config.tracking_uri,
                )
                iter_experiment_config = ExperimentConfig(**config_dict)
                logger.info(f"Loaded experiment config from MLflow run for '{conference_filter}'")
            except Exception as e:
                if prediction_config.feature_config_path:
                    logger.warning(
                        f"Could not load config from MLflow ({e}); "
                        f"falling back to {prediction_config.feature_config_path}"
                    )
                    fallback_path = PROJECT_ROOT / prediction_config.feature_config_path
                    base_config = load_experiment_config(fallback_path)
                    iter_experiment_config = base_config.model_copy(deep=True)
                    iter_experiment_config.filters.conference_filter = conference_filter
                else:
                    raise

            upcoming_with_features = build_features_for_prediction(
                upcoming_games=upcoming_games,
                historical_features=historical_features,
                experiment_config=iter_experiment_config,
                conference_filter=conference_filter,
            )

            if upcoming_with_features.empty:
                logger.warning(f"No {conference_filter}-conference games found, skipping")
                continue

            data_config = iter_experiment_config.data
            target_column = data_config.target_column
            X = prepare_features_for_model(
                df=upcoming_with_features,
                experiment_config=iter_experiment_config,
                target_column=target_column,
                conference_filter=conference_filter,
            )

            logger.info(f"Loading model for '{conference_filter}' from MLflow: {model_uri}")
            model, feature_names = _load_model(model_uri, mlflow_config.tracking_uri)

            X_aligned = _align_features(
                X,
                feature_names,
                allow_missing=prediction_config.allow_missing_features,
            )

            logger.info(f"Making predictions for '{conference_filter}'-conference games")
            predictions, probabilities = _predict(model, X_aligned)

            metadata_columns = iter_experiment_config.feature_engineering.metadata_columns
            part = _collect_metadata(
                upcoming_games, upcoming_with_features, metadata_columns
            )
            part["conference_filter"] = conference_filter
            part["prediction"] = predictions
            if probabilities is not None:
                part["home_win_probability"] = probabilities

            results_parts.append(part)

        if not results_parts:
            logger.warning("No predictions generated")
            return

        # Which date column survives depends on the experiment's metadata_columns.
        output = pd.concat(results_parts)
        sort_column = next(
            (col for col in ("gameDate", "gameDateOnlyStr") if col in output.columns),
            None,
        )
        if sort_column:
            output = output.sort_values(sort_column)

        output_path = PROJECT_ROOT / prediction_config.paths.output
        n_written = upsert_predictions(output, output_path)
        logger.info(f"Predicted {len(output)} games; {n_written} rows now in {output_path}")

        # Log metrics and artifacts
        tracker.log_params({"n_games": len(output)})
        tracker.log_artifact(str(output_path), artifact_path="predictions")

        logger.info("Prediction complete!")
