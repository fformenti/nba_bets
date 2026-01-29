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
from src.config import PROJECT_ROOT
from src.etl.features import create_features_tables, merge_features
from src.ml.config.loader import load_experiment_config, load_prediction_config
from src.ml.config.schema import ExperimentConfig, PredictionConfig

from src.ml.scripts.train_classifier import load_and_validate_data

from src.ml.prediction.io import load_upcoming_games

from src.ml.features.engineering import (
    create_conference_delta,
    get_home_conference_vs_away_conference_record,
    create_delta_features,
)
from src.ml.tracking import MLflowTracker

logger = get_logger(__name__)


def _load_model(
    config: PredictionConfig,
) -> Tuple[object, Optional[list[str]]]:
    """Load model from MLflow."""
    if not config.model_uri:
        raise ValueError(
            "model_uri is required. Please specify an MLflow model URI "
            "(e.g., 'models:/nba_classification_random_forest/Production' or 'runs:/<run_id>/model')"
        )

    if config.tracking_uri:
        mlflow.set_tracking_uri(config.tracking_uri)

    model = mlflow.sklearn.load_model(config.model_uri)
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


def setup_mlflow_tracking(tracking_uri: Optional[str] = None) -> None:
    """
    Setup MLflow tracking URI.

    Parameters
    ----------
    tracking_uri : str, optional
        Custom tracking URI. If None, uses default SQLite database.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI set to: {tracking_uri}")
    else:
        mlflow_db_path = PROJECT_ROOT / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
        logger.info(f"MLflow tracking URI set to: sqlite:///{mlflow_db_path}")


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
    columns_to_drop = ["arenaCity", "arenaName"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Set placeholder values for upcoming games
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
    add_conference_delta: bool,
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
        Conference filter type
    add_conference_delta : bool
        Whether to add conference delta features

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
    lags = feat_eng_config.lags
    location_lags = feat_eng_config.location_lags

    # Create feature tables
    logger.info("Creating feature tables")
    create_features_tables(historical_combined, lags, location_lags)

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
        lags=lags,
        location_lags=location_lags,
    )

    # Add conference-specific features
    if conference_filter == "different":
        logger.info("Adding home conference vs away conference record features")
        upcoming_with_features = get_home_conference_vs_away_conference_record(
            upcoming_with_features
        )

    if add_conference_delta:
        logger.info("Adding conference delta features")
        upcoming_with_features = create_conference_delta(upcoming_with_features)

    return upcoming_with_features


def prepare_features_for_model(
    df: pd.DataFrame,
    experiment_config: ExperimentConfig,
    target_column: str,
) -> pd.DataFrame:
    """
    Prepare features DataFrame by excluding metadata and target columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with all columns
    experiment_config : ExperimentConfig
        Experiment configuration
    target_column : str
        Target column name

    Returns
    -------
    pd.DataFrame
        Features DataFrame ready for model prediction
    """
    feat_eng_config = experiment_config.feature_engineering
    metadata_columns = feat_eng_config.metadata_columns
    originally_enriched_columns = feat_eng_config.originally_enriched_columns
    exclude_columns = feat_eng_config.exclude_columns

    # Combine all columns to exclude
    columns_to_exclude = (
        metadata_columns
        + originally_enriched_columns
        + [target_column]
        + (exclude_columns or [])
    )

    # Drop excluded columns
    X = df.drop(columns=[col for col in columns_to_exclude if col in df.columns])

    logger.info(f"Prepared {len(X.columns)} features for model prediction")
    logger.debug(f"Feature columns: {list(X.columns)}")

    return X


def main(
    config_path: Optional[Path] = None,
    experiment_name: str = "nba_bets_predictions",
):
    """
    Main prediction function.

    Parameters
    ----------
    config_path : Path, optional
        Path to prediction configuration YAML file. If None, uses default configuration.
    experiment_name : str, default='nba_bets_predictions'
        MLflow experiment name for tracking
    """
    # Setup logging
    setup_logging(level="INFO")

    # Load prediction configuration
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "predict_upcoming.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Prediction config file not found: {config_path}")

    prediction_config = load_prediction_config(config_path)
    logger.info(f"Loaded prediction configuration from {config_path}")

    # Setup MLflow tracking
    setup_mlflow_tracking(prediction_config.tracking_uri)

    # Load experiment configuration for feature engineering
    feature_config_path = PROJECT_ROOT / prediction_config.feature_config_path
    if not feature_config_path.exists():
        raise FileNotFoundError(f"Feature config file not found: {feature_config_path}")

    experiment_config = load_experiment_config(feature_config_path)
    logger.info(f"Loaded experiment configuration from {feature_config_path}")

    # Start MLflow tracking
    with MLflowTracker(
        experiment_name=experiment_name,
        run_name=config_path.stem if config_path else None,
        tracking_uri=prediction_config.tracking_uri,
        log_model=False,
    ) as tracker:
        # Log configuration
        tracker.log_params(
            {
                "model_uri": prediction_config.model_uri,
                "input_dir": prediction_config.input_dir,
                "output_path": prediction_config.output_path,
                "features_path": prediction_config.features_path,
                "feature_config_path": prediction_config.feature_config_path,
                "conference_filter": prediction_config.conference_filter,
                "add_conference_delta": prediction_config.add_conference_delta,
                "allow_missing_features": prediction_config.allow_missing_features,
            }
        )
        # Data loading
        data_config = prediction_config.data
        target_column = data_config.target_column
        date_column = data_config.date_column

        # Load upcoming games
        input_dir = PROJECT_ROOT / prediction_config.input_dir
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
        features_path = PROJECT_ROOT / prediction_config.features_path
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

        # Build features
        upcoming_with_features = build_features_for_prediction(
            upcoming_games=upcoming_games,
            historical_features=historical_features,
            experiment_config=experiment_config,
            conference_filter=prediction_config.conference_filter,
            add_conference_delta=prediction_config.add_conference_delta,
        )

        if upcoming_with_features.empty:
            logger.warning("No games remaining after feature building and filtering")
            return

        # Prepare features for model
        data_config = experiment_config.data
        target_column = data_config.target_column
        X = prepare_features_for_model(
            df=upcoming_with_features,
            experiment_config=experiment_config,
            target_column=target_column,
        )

        # Load model
        logger.info(f"Loading model from MLflow: {prediction_config.model_uri}")
        model, feature_names = _load_model(prediction_config)

        # Check for missing values before alignment
        missing_before = X.isna().sum()
        if missing_before.sum() > 0:
            columns_with_missing_before = missing_before[missing_before > 0]
            logger.info(
                f"Missing values in features before alignment:\n{columns_with_missing_before.to_dict()}"
            )
            rows_with_missing_before = X.isna().any(axis=1).sum()
            logger.info(
                f"Rows with missing values before alignment: {rows_with_missing_before} out of {len(X)}"
            )

        # Align features to match model expectations
        logger.info("Aligning features to model requirements")
        X_aligned = _align_features(
            X,
            feature_names,
            allow_missing=prediction_config.allow_missing_features,
        )

        # Make predictions
        logger.info("Making predictions")
        predictions, probabilities = _predict(model, X_aligned)

        # Prepare output
        metadata_columns = experiment_config.feature_engineering.metadata_columns
        output = upcoming_games[metadata_columns].copy()
        output["prediction"] = predictions

        if probabilities is not None:
            output["home_win_probability"] = probabilities

        # Save predictions
        output_path = PROJECT_ROOT / prediction_config.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False)
        logger.info(f"Saved predictions to {output_path}")

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
