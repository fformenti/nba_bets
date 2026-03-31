"""
Refactored prediction script for classification models.

This script demonstrates best practices:
- Configuration management via YAML
- Proper logging instead of print statements
- Modular feature engineering
- Input validation and error handling
- Clean separation of concerns
"""

from pathlib import Path
from typing import Optional, Tuple

import mlflow
import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logging, get_logger
from src.config.paths import PROJECT_ROOT
from src.etl.features.aggregator import create_features_tables, merge_features
from src.ml.config.loader import load_experiment_config, load_prediction_config
from src.ml.config.schema import ExperimentConfig
from src.ml.scripts.train_classifier import load_and_validate_data

from src.ml.prediction.io import load_upcoming_games

from src.ml.features.engineering import (
    create_delta_features,
    apply_conference_features,
    resolve_feature_columns,
)
from src.ml.tracking.mlflow_tracker import (
    MLflowTracker,
    load_experiment_config_from_model_uri,
)
from src.etl.utils.common import get_nba_season

logger = get_logger(__name__)


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
        aligned[col] = (
            np.nan
        )  # Use numpy NaN instead of pd.NA for sklearn compatibility

    # Reorder columns to match feature_names
    aligned = aligned[feature_names]

    # Check for missing values and log details
    missing_mask = aligned.isna().any(axis=1)
    if missing_mask.any():
        n_rows_with_missing = missing_mask.sum()
        logger.info(
            f"Rows with missing values: {n_rows_with_missing} out of {len(aligned)}"
        )

        # Log which columns have missing values
        missing_per_column = aligned.isna().sum()
        columns_with_missing = missing_per_column[missing_per_column > 0]
        if len(columns_with_missing) > 0:
            logger.info(
                f"Features with missing values:\n{columns_with_missing.to_dict()}"
            )

        # Log row indices with missing values
        rows_with_missing = aligned[missing_mask].index.tolist()
        logger.debug(f"Row indices with missing values: {rows_with_missing}")
    else:
        logger.info("No missing values found in aligned features")

    return aligned


def _predict(
    model: object, features: pd.DataFrame
) -> Tuple[pd.Series, Optional[pd.Series]]:
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


def fix_upcoming_games_cols(
    upcoming_games: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare upcoming games DataFrame with required columns.

    Parameters
    ----------
    upcoming : pd.DataFrame
        Raw upcoming games DataFrame
    teams_history : pd.DataFrame
        Teams history DataFrame with conferences

    Returns
    -------
    pd.DataFrame
        Prepared upcoming games DataFrame
    """
    logger.info(f"Preparing {len(upcoming_games)} upcoming games")

    df = upcoming_games.copy()
    if "gameDateOnlyStr" not in df.columns and "gameDate" in df.columns:
        df["gameDateOnlyStr"] = pd.to_datetime(
            df["gameDate"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
    if "season" not in df.columns and "gameDate" in df.columns:
        df["season"] = pd.to_datetime(df["gameDate"], errors="coerce").apply(
            get_nba_season
        )
    columns_to_drop = ["arenaCity", "arenaName"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Set placeholder values for upcoming games
    df["postponed"] = 0
    df["overtimes"] = None
    df["winner"] = 0
    df["homeScore"] = 0
    df["awayScore"] = 0
    df["arenaId"] = None
    df["attendance"] = None
    df["gameType"] = "Regular Season"

    return df


def filter_by_conference(
    df: pd.DataFrame,
    conference_filter: str,
) -> pd.DataFrame:
    """
    Filter games by conference matchup type.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with conference columns
    conference_filter : str
        Filter type: 'different', 'same', or 'all'

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    if conference_filter == "different":
        filtered = df[df["hometeamConference"] != df["awayteamConference"]].copy()
        logger.info(f"Filtered to {len(filtered)} games (different conferences)")
    elif conference_filter == "same":
        filtered = df[df["hometeamConference"] == df["awayteamConference"]].copy()
        logger.info(f"Filtered to {len(filtered)} games (same conference)")
    else:
        filtered = df.copy()
        logger.info(f"No conference filter applied ({len(filtered)} games)")

    return filtered


def build_features_for_prediction(
    upcoming_games: pd.DataFrame,
    historical_features: pd.DataFrame,
    experiment_config: ExperimentConfig,
    conference_filter: str,
) -> pd.DataFrame:
    """
    Build features for upcoming games prediction.

    Parameters
    ----------
    upcoming_games : pd.DataFrame
        Prepared upcoming games DataFrame
    historical_features : pd.DataFrame
        Historical features DataFrame
    experiment_config : ExperimentConfig
        Experiment configuration with feature engineering settings
    conference_filter : str
        Conference filter type ('same', 'different', or 'all').
        Determines which conference features to create:
        - 'same': No conference features
        - 'different': Conference vs conference record features
        - 'all': Conference delta feature (0.0 for same conference)

    Returns
    -------
    pd.DataFrame
        DataFrame with features ready for prediction
    """
    logger.info("Building features for prediction")

    # Combine historical and upcoming games for feature calculation
    historical_combined = pd.concat([historical_features, upcoming_games])

    # Get feature engineering configuration
    feat_eng_config = experiment_config.feature_engineering
    # Create feature tables (uses backward-compatible property accessors)
    logger.info("Creating feature tables")
    create_features_tables(
        historical_combined,
        feat_eng_config.record_lags,
        feat_eng_config.point_differential_lags,
        feat_eng_config.location_lags,
        feat_eng_config.distances_lags,
        feat_eng_config.sos_lags,
        sos_adj_alpha=feat_eng_config.sos_adj_alpha,
        sos_adj_location_lags=feat_eng_config.features.sos_adj_record.location_lags,
    )

    # Merge features for upcoming games
    upcoming_with_features = merge_features(upcoming_games)

    # Drop columns that are not needed for feature engineering
    # These columns are added for merge_features but shouldn't be in final features
    columns_to_drop_after_merge = ["winner", "winnerteamConference"]
    upcoming_with_features = upcoming_with_features.drop(
        columns=[
            col
            for col in columns_to_drop_after_merge
            if col in upcoming_with_features.columns
        ]
    )

    # Apply conference filter
    upcoming_with_features = filter_by_conference(
        upcoming_with_features, conference_filter
    )

    # Create delta features
    logger.info("Creating delta features")
    upcoming_with_features = create_delta_features(
        upcoming_with_features,
        feat_eng_config.features,
    )

    # Apply conference-specific features based on filter type
    # This matches the training logic exactly - conference_filter determines features
    upcoming_with_features = apply_conference_features(
        upcoming_with_features, conference_filter
    )

    return upcoming_with_features


def prepare_features_for_model(
    df: pd.DataFrame,
    experiment_config: ExperimentConfig,
    target_column: str,
    conference_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Prepare features DataFrame for model prediction.

    In inclusion mode, only columns declared in the experiment config are kept.
    In exclusion mode (legacy), metadata and intermediate columns are dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with all columns
    experiment_config : ExperimentConfig
        Experiment configuration
    target_column : str
        Target column name
    conference_filter : str, optional
        Conference filter used to resolve conference feature columns. Falls back
        to experiment_config.filters.conference_filter when not provided.

    Returns
    -------
    pd.DataFrame
        Features DataFrame ready for model prediction
    """
    feat_eng_config = experiment_config.feature_engineering

    if feat_eng_config.selection_mode == "inclusion":
        cf = conference_filter or experiment_config.filters.conference_filter
        feature_columns = resolve_feature_columns(feat_eng_config.features, cf)
        available = [c for c in feature_columns if c in df.columns]
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
        metadata_columns = feat_eng_config.metadata_columns
        intermediate_columns = feat_eng_config.intermediate_columns
        exclude_columns = feat_eng_config.exclude_columns
        columns_to_exclude = (
            metadata_columns
            + intermediate_columns
            + [target_column]
            + (exclude_columns or [])
        )
        X = df.drop(columns=[col for col in columns_to_exclude if col in df.columns])

    logger.info(f"Prepared {len(X.columns)} features for model prediction")
    logger.debug(f"Feature columns: {list(X.columns)}")

    return X


def main(
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
        config_path = PROJECT_ROOT / "configs" / "predict_classifier.yaml"

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
                logger.info(
                    f"Loaded experiment config from MLflow run for '{conference_filter}'"
                )
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
                logger.warning(
                    f"No {conference_filter}-conference games found, skipping"
                )
                continue

            data_config = iter_experiment_config.data
            target_column = data_config.target_column
            X = prepare_features_for_model(
                df=upcoming_with_features,
                experiment_config=iter_experiment_config,
                target_column=target_column,
                conference_filter=conference_filter,
            )

            logger.info(
                f"Loading model for '{conference_filter}' from MLflow: {model_uri}"
            )
            model, feature_names = _load_model(model_uri, mlflow_config.tracking_uri)

            X_aligned = _align_features(
                X,
                feature_names,
                allow_missing=prediction_config.allow_missing_features,
            )

            logger.info(
                f"Making predictions for '{conference_filter}'-conference games"
            )
            predictions, probabilities = _predict(model, X_aligned)

            metadata_columns = (
                iter_experiment_config.feature_engineering.metadata_columns
            )
            part = upcoming_games.loc[
                upcoming_with_features.index, metadata_columns
            ].copy()
            part["conference_filter"] = conference_filter
            part["prediction"] = predictions
            if probabilities is not None:
                part["home_win_probability"] = probabilities

            results_parts.append(part)

        if not results_parts:
            logger.warning("No predictions generated")
            return

        output = pd.concat(results_parts).sort_values("gameDate")

        # Save predictions (append if file exists, otherwise write with header)
        output_path = PROJECT_ROOT / prediction_config.paths.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not output_path.exists()
        output.to_csv(output_path, index=False, mode="a", header=write_header)
        logger.info(f"Saved {len(output)} predictions to {output_path}")

        # Log metrics and artifacts
        tracker.log_params({"n_games": len(output)})
        tracker.log_artifact(str(output_path), artifact_path="predictions")

        logger.info("Prediction complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict outcomes for upcoming NBA games"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to prediction configuration YAML file",
    )
    args = parser.parse_args()

    main(config_path=args.config)
