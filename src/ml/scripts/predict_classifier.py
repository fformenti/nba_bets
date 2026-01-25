from __future__ import annotations


import mlflow
import pandas as pd
from pathlib import Path

# pd.set_option("display.max_rows", 100)
# pd.set_option("display.max_columns", 100)

from src.config import PROJECT_ROOT
from src.etl.ingestion.raw_games import get_nba_season
from src.ml.config.loader import load_experiment_config

from src.ml.tracking import MLflowTracker
from src.utils.logging_config import setup_logging, get_logger
from src.etl.transformation import add_conference
from src.etl.ingestion.teams_history import create_teams_history_table
from src.config import (
    LOCAL_TEAMS_HISTORY_CITIES_CONFERENCES_PATH,
    LOCAL_GAMES_FEATURES_PATH,
)

from src.etl.features import create_features_tables, merge_features

from src.ml.features.engineering import (
    create_conference_delta,
    get_home_conference_vs_away_conference_record,
    create_delta_features,
)

from src.ml.prediction.config import load_prediction_config
from src.ml.prediction.io import load_historical_features, load_upcoming_games

from src.ml.prediction.pipeline import _load_model
from src.ml.prediction.pipeline import _predict
from src.ml.prediction.pipeline import _align_features


# add nba_bets to python path
import sys

sys.path.append("/Users/felipeformenti/dev/fformenti/nba_bets")

# Set MLflow tracking URI to use the root mlflow.db
# This ensures we use the project root database instead of creating one in src/notebooks/
logger = get_logger(__name__)

setup_logging(level="INFO")

mlflow_db_path = PROJECT_ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
print(f"MLflow tracking URI set to: sqlite:///{mlflow_db_path}")


config, config_path = None, None
if config is None:
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "predict_upcoming.yaml"
    config = load_prediction_config(config_path)

training_config_path = PROJECT_ROOT / "configs" / "my_experiment.yaml"
training_config = load_experiment_config(training_config_path)


upcoming = load_upcoming_games(
    PROJECT_ROOT / config.input_dir, max_files=config.max_files
)

historical = load_historical_features(PROJECT_ROOT / config.features_path)
upcoming_seasons = upcoming["gameDate"].apply(get_nba_season).dropna().unique().tolist()
if upcoming_seasons:
    historical = historical[historical["season"].isin(upcoming_seasons)].copy()

feature_config = load_experiment_config(PROJECT_ROOT / config.feature_config_path)

teams_history_path = "/Users/felipeformenti/dev/fformenti/nba_bets/data/raw/historical/TeamsHistoriesConferenceNBA.csv"
current_season_year = 2024

teams_history = create_teams_history_table(
    input_file=str(teams_history_path),
    output_file=str(LOCAL_TEAMS_HISTORY_CITIES_CONFERENCES_PATH),
    current_season_year=current_season_year,
)

upcoming.drop(columns=["arenaCity", "arenaName"], inplace=True)
# add add season to upcoming
upcoming["season"] = upcoming["gameDate"].apply(get_nba_season)
# add column gameDateOnlyStr to upcoming
upcoming["gameDateOnlyStr"] = upcoming["gameDate"].dt.strftime("%Y-%m-%d")
# add column hometeamConference to upcoming
upcoming["winner"] = 0
upcoming = add_conference(upcoming, teams_history)

# upcoming.drop(columns=["winner", "winnerteamConference"], inplace=True)
upcoming["homeScore"] = 0
upcoming["awayScore"] = 0
upcoming["arenaId"] = None
upcoming["attendance"] = None
upcoming["gameType"] = "Regular Season"


# append upcoming to historical
historical_new = pd.concat([historical, upcoming])

location_lags = feature_config.feature_engineering.location_lags
lags = feature_config.feature_engineering.lags

create_features_tables(historical_new, lags, location_lags)
upcoming_games = merge_features(upcoming)

# model configuration
only_different_conferences = True
only_same_conferences = False
within_and_across_conferences = False

# Subset only games where teams are from different conferences
if only_different_conferences:
    upcoming_games = upcoming_games[
        upcoming_games["hometeamConference"] != upcoming_games["awayteamConference"]
    ]
if only_same_conferences:
    upcoming_games = upcoming_games[
        upcoming_games["hometeamConference"] == upcoming_games["awayteamConference"]
    ]

experiment_config = load_experiment_config(config_path)
data_config = experiment_config.data
# Data loading
data_path = (
    Path(data_config.path) if data_config.path else Path(LOCAL_GAMES_FEATURES_PATH)
)
target_column = data_config.target_column
date_column = data_config.date_column


metadata_columns = feature_config.feature_engineering.metadata_columns
originally_enriched_columns = (
    feature_config.feature_engineering.originally_enriched_columns
)


# upcoming_games["homeScore"] = 0
# upcoming_games["awayScore"] = 0
upcoming_games.drop(columns=metadata_columns, inplace=True)


df = create_delta_features(upcoming_games, lags, location_lags)
# print(df.columns)

# Combine conference features (not necessary if only within conferences)
if only_different_conferences:
    # This will create the features: home_conference_vs_away_conference_record and games_played_at_home_conference
    df = get_home_conference_vs_away_conference_record(df)
if within_and_across_conferences:
    # This will create the feature: conference_diff_east_pct
    df = create_conference_delta(df)

# Select features for model
exclude_cols = metadata_columns + originally_enriched_columns + [target_column]

X = df.drop(columns=[col for col in exclude_cols if col in df.columns])

model, feature_names = _load_model(config)
# Align features to match the order expected by the model (from training)
# scikit-learn uses column ORDERING, not column names when making predictions
X_aligned = _align_features(X, feature_names, allow_missing=False)
predictions, probabilities = _predict(model, X_aligned)

# output = build_result.metadata.copy()
output = upcoming[metadata_columns].copy()
output["prediction"] = predictions
if probabilities is not None:
    output["home_win_probability"] = probabilities

output_path = PROJECT_ROOT / config.output_path
output_path.parent.mkdir(parents=True, exist_ok=True)
output.to_csv(output_path, index=False)
logger.info("Saved predictions to %s", output_path)

with MLflowTracker(
    experiment_name=config.experiment_name,
    run_name="upcoming_predictions",
    tracking_uri=config.tracking_uri,
    log_model=False,
) as tracker:
    tracker.log_params(
        {
            "model_uri": config.model_uri,
            "input_dir": config.input_dir,
            "output_path": config.output_path,
            "n_games": len(output),
        }
    )
    tracker.log_artifact(str(output_path), artifact_path="predictions")
