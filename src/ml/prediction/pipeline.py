# """Prediction pipeline for upcoming NBA games using MLflow."""

# from __future__ import annotations

# from typing import Optional, Tuple

# import mlflow
# import pandas as pd

# from src.config import PROJECT_ROOT, TEAMS_CITIES_CONFERENCE_HISTORY_PROCESSED_PATH
# from src.utils.logging_config import get_logger
# from src.ml.config.loader import load_experiment_config
# from src.etl.ingestion.raw_games import get_nba_season
# from src.etl.ingestion.teams_history import load_teams_history_table
# from src.etl.transformation import add_conference

# from src.etl.features import create_features_tables, merge_features
# from .config import PredictionConfig, load_prediction_config

# # from .feature_builder import build_features_for_upcoming
# from .io import load_historical_features, load_upcoming_games


# logger = get_logger(__name__)


# def _load_model(
#     config: PredictionConfig,
# ) -> Tuple[object, Optional[list[str]]]:
#     """Load model from MLflow."""
#     if not config.model_uri:
#         raise ValueError(
#             "model_uri is required. Please specify an MLflow model URI "
#             "(e.g., 'models:/nba_classification_random_forest/Production' or 'runs:/<run_id>/model')"
#         )

#     if config.tracking_uri:
#         mlflow.set_tracking_uri(config.tracking_uri)

#     model = mlflow.sklearn.load_model(config.model_uri)
#     feature_names = getattr(model, "feature_names_in_", None)
#     return model, list(feature_names) if feature_names is not None else None


# def _align_features(
#     features: pd.DataFrame,
#     feature_names: Optional[list[str]],
#     allow_missing: bool,
# ) -> pd.DataFrame:
#     if feature_names is None:
#         return features

#     missing = [col for col in feature_names if col not in features.columns]
#     if missing and not allow_missing:
#         raise ValueError(f"Missing required features: {missing}")

#     aligned = features.copy()
#     for col in missing:
#         aligned[col] = pd.NA
#     return aligned[feature_names]


# def _predict(
#     model: object, features: pd.DataFrame
# ) -> Tuple[pd.Series, Optional[pd.Series]]:
#     predictions = model.predict(features)
#     preds = pd.Series(predictions, index=features.index)

#     proba = None
#     if hasattr(model, "predict_proba"):
#         proba_raw = model.predict_proba(features)
#         if hasattr(model, "classes_"):
#             classes = list(model.classes_)
#             if 1 in classes:
#                 class_index = classes.index(1)
#                 proba = pd.Series(proba_raw[:, class_index], index=features.index)
#             else:
#                 proba = pd.Series(proba_raw.max(axis=1), index=features.index)
#         else:
#             proba = pd.Series(proba_raw.max(axis=1), index=features.index)

#     return preds, proba


# def append_upcoming_to_historical():
#     config, config_path = None, None
#     if config is None:
#         if config_path is None:
#             config_path = PROJECT_ROOT / "configs" / "predict_upcoming.yaml"
#         config = load_prediction_config(config_path)
#     upcoming = load_upcoming_games(
#         PROJECT_ROOT / config.input_dir, max_files=config.max_files
#     )

#     historical = load_historical_features(PROJECT_ROOT / config.features_path)
#     upcoming_seasons = (
#         upcoming["gameDate"].apply(get_nba_season).dropna().unique().tolist()
#     )
#     if upcoming_seasons:
#         historical = historical[historical["season"].isin(upcoming_seasons)].copy()

#     feature_config = load_experiment_config(PROJECT_ROOT / config.feature_config_path)

#     teams_history_path = TEAMS_CITIES_CONFERENCE_HISTORY_PROCESSED_PATH
#     teams_history = load_teams_history_table(processed_file=str(teams_history_path))

#     upcoming.drop(columns=["arenaCity", "arenaName"], inplace=True)
#     # add add season to upcoming
#     upcoming["season"] = upcoming["gameDate"].apply(get_nba_season)
#     # add column gameDateOnlyStr to upcoming
#     upcoming["gameDateOnlyStr"] = upcoming["gameDate"].dt.strftime("%Y-%m-%d")
#     # add column hometeamConference to upcoming
#     upcoming["winner"] = 0
#     upcoming = add_conference(upcoming, teams_history)

#     # upcoming.drop(columns=["winner", "winnerteamConference"], inplace=True)
#     # upcoming["arenaId"] = None
#     # upcoming["attendance"] = None

#     historical_new = pd.concat([historical, upcoming])

#     location_lags = feature_config.feature_engineering.location_lags
#     lags = feature_config.feature_engineering.location_lags

#     create_features_tables(historical_new, lags, location_lags)

#     upcoming.drop(columns=["winner", "winnerteamConference"], inplace=True)
#     upcoming["arenaId"] = None
#     upcoming["attendance"] = None

#     final_features = merge_features(upcoming)
#     return final_features


# def run_prediction_pipeline(
#     config_path: Optional[Path] = None,
#     config: Optional[PredictionConfig] = None,
# ) -> pd.DataFrame:
#     """
#     Run the end-to-end prediction pipeline for upcoming games.
#     """
#     setup_logging(level="INFO")

#     if config is None:
#         if config_path is None:
#             config_path = PROJECT_ROOT / "configs" / "predict_upcoming.yaml"
#         config = load_prediction_config(config_path)

#     upcoming = load_upcoming_games(
#         PROJECT_ROOT / config.input_dir, max_files=config.max_files
#     )
#     if upcoming.empty:
#         logger.info("No upcoming games found to score.")
#         return pd.DataFrame()

#     historical = load_historical_features(PROJECT_ROOT / config.features_path)
#     upcoming_seasons = (
#         upcoming["gameDate"].apply(get_nba_season).dropna().unique().tolist()
#     )
#     if upcoming_seasons:
#         historical = historical[historical["season"].isin(upcoming_seasons)].copy()
#     feature_config = load_experiment_config(PROJECT_ROOT / config.feature_config_path)

#     build_result = build_features_for_upcoming(
#         upcoming=upcoming,
#         historical=historical,
#         config=feature_config,
#         conference_filter=config.conference_filter,
#         add_conference_delta=config.add_conference_delta,
#     )

#     model, feature_names = _load_model(config)
#     aligned_features = _align_features(
#         build_result.features, feature_names, config.allow_missing_features
#     )

#     predictions, probabilities = _predict(model, aligned_features)

#     output = build_result.metadata.copy()
#     output["prediction"] = predictions
#     if probabilities is not None:
#         output["home_win_probability"] = probabilities

#     output_path = PROJECT_ROOT / config.output_path
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     output.to_csv(output_path, index=False)
#     logger.info("Saved predictions to %s", output_path)

#     with MLflowTracker(
#         experiment_name=config.experiment_name,
#         run_name="upcoming_predictions",
#         tracking_uri=config.tracking_uri,
#         log_model=False,
#     ) as tracker:
#         tracker.log_params(
#             {
#                 "model_uri": config.model_uri,
#                 "input_dir": config.input_dir,
#                 "output_path": config.output_path,
#                 "n_games": len(output),
#             }
#         )
#         tracker.log_artifact(str(output_path), artifact_path="predictions")

#     return output
