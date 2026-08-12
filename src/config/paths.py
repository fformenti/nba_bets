"""
Path constants for data files and directories.

All paths are relative to PROJECT_ROOT for portability.
"""

from pathlib import Path
from src.config.constants import LEAGUE_SCHEDULE_FILE

# Project root directory
script_dir = Path(__file__).parent
PROJECT_ROOT = script_dir.parent.parent

# ===== Config YAML (single source of truth for CLI defaults) =====
CONFIGS_DIR = PROJECT_ROOT / "configs"
CONFIGS_TRAIN_DIR = CONFIGS_DIR / "train"
# Aligns with Makefile TRAIN_CONFIG ?= xgboost / make train.
# Swap to all_models.yaml (same features and splits, four model families) with
# `make train TRAIN_CONFIG=all_models`.
DEFAULT_TRAIN_CLASSIFIER_CONFIG_PATH = CONFIGS_TRAIN_DIR / "xgboost.yaml"
# Every train config _includes this one and none override `splitting`, so it is
# the single declaration of the train/validation/test season boundaries. Read by
# the Polymarket slug builder, which must cover exactly the test seasons, and by
# the trainer, for which model hyperparameters were explicitly declared.
TRAIN_DEFAULTS_CONFIG_PATH = CONFIGS_TRAIN_DIR / "_defaults.yaml"
DEFAULT_FEATURES_CONFIG_PATH = CONFIGS_DIR / "features.yaml"
# Two directories because they hold two different schemas, not two flavours of
# one: configs/train/ is ExperimentConfig (features, splits, sklearn models),
# configs/train_llm/ is LLMTrainingConfig (base model, LoRA, quantization), and
# each has its own loader. configs/train/llm_features.yaml is an ExperimentConfig
# despite the name — it defines the columns the LLM dataset is built from.
CONFIGS_TRAIN_LLM_DIR = CONFIGS_DIR / "train_llm"
# Aligns with Makefile LLM_CONFIG ?= llama31_8b_qlora / make train-llm
DEFAULT_TRAIN_LLM_CONFIG_PATH = CONFIGS_TRAIN_LLM_DIR / "llama31_8b_qlora.yaml"
# Aligns with Makefile PREDICTION_CONFIG ?= predict_classifier / make predict-upcoming
CONFIGS_PREDICT_DIR = CONFIGS_DIR / "predict"
DEFAULT_PREDICT_CONFIG_PATH = CONFIGS_PREDICT_DIR / "predict_classifier.yaml"

# ===== Base data directories =====
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INGESTED_DIR = DATA_DIR / "ingested"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"

# ===== Subdirectories =====
REGULAR_SEASON_DIR = PROCESSED_DIR / "regular_season"

# ===== Raw =====
RAW_HISTORICAL_DIR = RAW_DIR / "historical"
RAW_GAMES_PATH = RAW_HISTORICAL_DIR / "games" / "Games.csv"
LEAGUE_SCHEDULE_PATH = RAW_HISTORICAL_DIR / LEAGUE_SCHEDULE_FILE

# ===== Reference inputs =====
# Generated once by an external service rather than derived from anything in
# this repo, so they are *inputs* and live under raw/ with the other inputs.
# Nothing in the pipeline can rebuild them offline: losing them stops the ETL at
# its first step. See docs/PIPELINE_AUDIT.md and the builders in
# src/etl/reference/.
RAW_REFERENCE_DIR = RAW_DIR / "reference"
# teamId × season → city/state. Built by `make build-teams-locations` (OpenAI).
TEAMS_LOCATIONS_REFERENCE_PATH = (
    RAW_REFERENCE_DIR / "TeamsHistoriesLocationsNBALookUpTable.csv"
)
# City-pair travel distances. Built by `make build-distances-table`
# (Serper + OpenAI). Holds distinct pairs only — a team staying in the same city
# has no row here by construction, which is why an unmatched same-city pair
# means zero travel rather than missing data.
LOCATIONS_DISTANCES_PATH = RAW_REFERENCE_DIR / "locations_distances.csv"

# ===== Handmade =====
# Conference added by hand using the NBA_TEAMS_HISTORY_PATH file
TEAMS_CITIES_CONFERENCE_HISTORY_HANDMADE_PATH = (
    RAW_HISTORICAL_DIR / "handmade" / "TeamsHistoriesConferenceNBA.csv"
)

# ===== Collected =====
# One directory per state a fetched game can be in. A game moves forward through
# them exactly once, so "where is this file?" answers "what happened to it?".
RAW_INCREMENTAL_DIR = RAW_DIR / "incremental"
RAW_INCREMENTAL_ARCHIVE_DIR = RAW_INCREMENTAL_DIR / "archive"
# Pending: emitted by the selector, no terminal result from the source yet.
UPCOMING_GAMES_DIR = RAW_INCREMENTAL_DIR / "upcoming_games"
# Inbox: a final score has been fetched but is not in the history table yet.
UPCOMING_GAMES_RESULTS_DIR = RAW_INCREMENTAL_DIR / "upcoming_games_results"
# Consumed: already folded into INGESTED_GAMES_UPDATED_HISTORY_PATH. Kept for
# audit only — nothing reads these back.
ARCHIVE_RESULTS_DIR = RAW_INCREMENTAL_ARCHIVE_DIR / "results"
# Parked: the source reported the game postponed. Watched against a refreshed
# schedule for a makeup date; never written to history as a played game.
POSTPONED_GAMES_DIR = RAW_INCREMENTAL_DIR / "postponed"
# Quarantine: the source never returned a terminal status. Held here so a single
# unanswerable game cannot block the rest of the pipeline. Needs a human.
UNRESOLVED_GAMES_DIR = RAW_INCREMENTAL_DIR / "unresolved"
# Hand-dropped outcomes read by the placeholder results source, until a real
# provider is chosen. See src/etl/collectors/results/placeholder_source.py.
MANUAL_RESULTS_DIR = RAW_INCREMENTAL_DIR / "manual_results"

# ===== Ingested =====
# The authoritative game history — the only ingested table, written by exactly
# two steps and read by everything downstream. `make ingest-raw-games` merges the
# raw historical archive into it (the archive wins any gameId both hold);
# `make append-game-results` adds the results inbox to it.
INGESTED_GAMES_UPDATED_HISTORY_PATH = INGESTED_DIR / "games_updated_history.csv"

# ===== Processed =====
TEAMS_CITIES_CONFERENCE_HISTORY_PROCESSED_PATH = (
    PROCESSED_DIR / "TeamsHistoriesConferenceNBALookUpTable.csv"
)
TEAMS_ARENA_PATH = PROCESSED_DIR / "teams_arena.csv"
POLYMARKET_TEAMS_ABV_PATH = PROCESSED_DIR / "polymarket_teams_abv.csv"
PROCESSED_LEAGUE_SCHEDULE_PATH = PROCESSED_DIR / "league_schedule.csv"
REGULAR_SEASON_GAMES_PATH = REGULAR_SEASON_DIR / "games.csv"
NON_POSITIVE_SCORE_PATH = REGULAR_SEASON_DIR / "non_positive_score.csv"

# Feature tables paths
REGULAR_SEASON_FEATURES_DIR = PROCESSED_DIR / "regular_season" / "features"
REGULAR_SEASON_GAMES_FEATURES_PATH = REGULAR_SEASON_DIR / "games_features.csv"
TEAMS_HOME_RECORDS_PATH = REGULAR_SEASON_FEATURES_DIR / "teams_home_record.csv"
TEAMS_AWAY_RECORDS_PATH = REGULAR_SEASON_FEATURES_DIR / "teams_away_record.csv"
TEAMS_RECORDS_PATH = REGULAR_SEASON_FEATURES_DIR / "teams_records.csv"
TEAMS_HOME_PTS_DIFF_PATH = REGULAR_SEASON_FEATURES_DIR / "teams_home_pts_diff.csv"
TEAMS_AWAY_PTS_DIFF_PATH = REGULAR_SEASON_FEATURES_DIR / "teams_away_pts_diff.csv"
TEAMS_PTS_DIFF_PATH = REGULAR_SEASON_FEATURES_DIR / "teams_pts_diff.csv"
EAST_WEST_RECORDS_PATH = REGULAR_SEASON_FEATURES_DIR / "east_west_record.csv"
EAST_WEST_RECORDS_AT_EAST_PATH = (
    REGULAR_SEASON_FEATURES_DIR / "east_west_record_at_east.csv"
)
EAST_WEST_RECORDS_AT_WEST_PATH = (
    REGULAR_SEASON_FEATURES_DIR / "east_west_record_at_west.csv"
)
RESTED_DAYS_PATH = REGULAR_SEASON_FEATURES_DIR / "rested_days.csv"
TEAMS_DISTANCES_PATH = REGULAR_SEASON_FEATURES_DIR / "teams_distances.csv"
LAST_SEASON_RECORD_PATH = REGULAR_SEASON_FEATURES_DIR / "last_season_record.csv"
LAST_SEASON_HOME_RECORD_PATH = REGULAR_SEASON_FEATURES_DIR / "last_season_home_record.csv"
LAST_SEASON_AWAY_RECORD_PATH = REGULAR_SEASON_FEATURES_DIR / "last_season_away_record.csv"
LAST_SEASON_ADJ_RECORD_PATH = REGULAR_SEASON_FEATURES_DIR / "last_season_adj_record.csv"
LAST_SEASON_ADJ_HOME_RECORD_PATH = REGULAR_SEASON_FEATURES_DIR / "last_season_adj_home_record.csv"
LAST_SEASON_ADJ_AWAY_RECORD_PATH = REGULAR_SEASON_FEATURES_DIR / "last_season_adj_away_record.csv"
TEAMS_STREAKS_PATH = REGULAR_SEASON_FEATURES_DIR / "teams_streaks.csv"
TEAMS_SOS_PATH = REGULAR_SEASON_FEATURES_DIR / "teams_sos.csv"
TEAMS_SOS_ADJ_RECORD_PATH = REGULAR_SEASON_FEATURES_DIR / "teams_sos_adj_record.csv"
TEAMS_SOS_ADJ_HOME_RECORD_PATH = REGULAR_SEASON_FEATURES_DIR / "teams_sos_adj_home_record.csv"
TEAMS_SOS_ADJ_AWAY_RECORD_PATH = REGULAR_SEASON_FEATURES_DIR / "teams_sos_adj_away_record.csv"
TEAMS_HOME_NORM_PTS_DIFF_PATH = REGULAR_SEASON_FEATURES_DIR / "teams_home_norm_pts_diff.csv"
TEAMS_AWAY_NORM_PTS_DIFF_PATH = REGULAR_SEASON_FEATURES_DIR / "teams_away_norm_pts_diff.csv"
TEAMS_NORM_PTS_DIFF_PATH = REGULAR_SEASON_FEATURES_DIR / "teams_norm_pts_diff.csv"
TEAMS_GDS_PATH = REGULAR_SEASON_FEATURES_DIR / "teams_gds.csv"
TEAMS_GDS_HOME_PATH = REGULAR_SEASON_FEATURES_DIR / "teams_gds_home.csv"
TEAMS_GDS_AWAY_PATH = REGULAR_SEASON_FEATURES_DIR / "teams_gds_away.csv"
PLAYOFF_STANDINGS_PATH = REGULAR_SEASON_FEATURES_DIR / "playoff_standings.csv"

# Prediction builds the same tables, under the same filenames, from a much
# smaller frame: one season of history plus the upcoming slate. It gets its own
# directory because sharing one left the ETL's full-history tables truncated
# after every prediction run. Nothing reads these two directories except
# `merge_features`, which is told which one to use.
PREDICTION_FEATURES_DIR = PROCESSED_DIR / "prediction" / "features"

# Predictions
UPCOMING_GAMES_PREDICTIONS_PATH = PREDICTIONS_DIR / "upcoming_games_predictions.csv"
POLYMARKET_DAILY_BETS_DIR = PREDICTIONS_DIR / "daily_bets"
# Live model accuracy: predictions joined against games that have been played.
PREDICTION_SCORECARD_PATH = PREDICTIONS_DIR / "prediction_scorecard.csv"
PREDICTION_SCORED_GAMES_PATH = PREDICTIONS_DIR / "prediction_scored_games.csv"

# Polymarket analysis
GAME_SLUG_LOOKUP_PATH = PROCESSED_DIR / "game_slug_lookup.csv"

# ===== Outputs =====
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ENRICHED_GAMES_VIZ_DIR = OUTPUTS_DIR / "enriched_games" / "viz"
HOME_WIN_RATIO_BY_SEASON_PNG_PATH = ENRICHED_GAMES_VIZ_DIR / "home_win_ratio_by_season.png"

# LLM fine-tuning: checkpoints, downloaded resume snapshots and eval charts
LLM_OUTPUTS_DIR = OUTPUTS_DIR / "llm"
LLM_EVAL_OUTPUTS_DIR = LLM_OUTPUTS_DIR / "eval"


def project_relpath(path: Path) -> str:
    """Path relative to project root as a POSIX string (for YAML/config defaults)."""
    return path.relative_to(PROJECT_ROOT).as_posix()
