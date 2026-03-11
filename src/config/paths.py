"""
Path constants for data files and directories.

All paths are relative to PROJECT_ROOT for portability.
"""

from pathlib import Path
from src.config.constants import LEAGUE_SCHEDULE_FILE

# Project root directory
script_dir = Path(__file__).parent
PROJECT_ROOT = script_dir.parent.parent

# ===== Base data directories =====
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INGESTED_DIR = DATA_DIR / "ingested"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"

# ===== Subdirectories =====
REGULAR_SEASON_DIR = PROCESSED_DIR / "regular_season"
PLAYOFFS_DIR = PROCESSED_DIR / "playoffs"

# ===== Raw =====
RAW_HISTORICAL_DIR = RAW_DIR / "historical"
RAW_GAMES_PATH = RAW_HISTORICAL_DIR / "games" / "Games.csv"
RAW_DISTANCES_PATH = RAW_DIR / "distances_long.csv"
LEAGUE_SCHEDULE_PATH = RAW_HISTORICAL_DIR / LEAGUE_SCHEDULE_FILE
# downaloaded file
ALL_TEAMS_HISTORY_PATH = RAW_HISTORICAL_DIR / "TeamsHistories.csv"
# filter from file above filter for NBA teams
NBA_TEAMS_HISTORY_PATH = RAW_HISTORICAL_DIR / "TeamsHistoriesNBA.csv"

# ===== Handmade =====
# Conference added by hand using the NBA_TEAMS_HISTORY_PATH file
TEAMS_CITIES_CONFERENCE_HISTORY_HANDMADE_PATH = (
    RAW_HISTORICAL_DIR / "handmade" / "TeamsHistoriesConferenceNBA.csv"
)

# ===== Collected =====
RAW_INCREMENTAL_DIR = RAW_DIR / "incremental"
UPCOMING_GAMES_DIR = RAW_INCREMENTAL_DIR / "upcoming_games"
UPCOMING_GAMES_RESULTS_DIR = RAW_INCREMENTAL_DIR / "upcoming_games_results"

# ===== Ingested =====
INGESTED_GAMES_PATH = INGESTED_DIR / "historical" / "games.csv"
INGESTED_GAMES_UPDATED_HISTORY_PATH = INGESTED_DIR / "games_updated_history.csv"
POSTPONED_GAMES_PATH = INGESTED_DIR / "historical" / "postponed_games.csv"

# ===== Processed =====
TEAMS_CITIES_CONFERENCE_HISTORY_PROCESSED_PATH = (
    PROCESSED_DIR / "TeamsHistoriesConferenceNBALookUpTable.csv"
)
TEAMS_CITIES_LOCATIONS_HISTORY_PROCESSED_PATH = (
    PROCESSED_DIR / "TeamsHistoriesLocationsNBALookUpTable.csv"
)
LOCATIONS_DISTANCES_PATH = PROCESSED_DIR / "locations_distances.csv"

POLYMARKET_TEAMS_ABV_PATH = PROCESSED_DIR / "polymarket_teams_abv.csv"
PROCESSED_LEAGUE_SCHEDULE_PATH = PROCESSED_DIR / "league_schedule.csv"
REGULAR_SEASON_GAMES_PATH = REGULAR_SEASON_DIR / "games.csv"
NON_POSITIVE_SCORE_PATH = REGULAR_SEASON_DIR / "non_positive_score.csv"
PLAYOFFS_GAMES_PATH = PROCESSED_DIR / "playoffs" / "games"

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

# Predictions
UPCOMING_GAMES_PREDICTIONS_PATH = PREDICTIONS_DIR / "upcoming_games_predictions.csv"
POLYMARKET_DAILY_BETS_DIR = PREDICTIONS_DIR / "daily_bets"
