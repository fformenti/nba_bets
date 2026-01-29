"""
Path constants for data files and directories.

All paths are relative to PROJECT_ROOT for portability.
"""

from pathlib import Path

# Project root directory
script_dir = Path(__file__).parent
PROJECT_ROOT = script_dir.parent.parent

# Base data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INGESTED_DIR = DATA_DIR / "ingested"
PROCESSED_DIR = DATA_DIR / "processed"


# ===== Raw =====
RAW_HISTORICAL_DIR = RAW_DIR / "historical"
RAW_GAMES_PATH = RAW_HISTORICAL_DIR / "games" / "Games.csv"

# To do: Remove hardcoded "LeagueSchedule25_26.csv" from here. Find a better place for it
LEAGUE_SCHEDULE_PATH = RAW_HISTORICAL_DIR / "LeagueSchedule25_26.csv"
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
RAW_INCREMENTAL_ARCHIVE_DIR = RAW_INCREMENTAL_DIR / "archive"
UPCOMING_GAMES_DIR = RAW_INCREMENTAL_DIR / "upcoming_games"
UPCOMING_GAMES_RESULTS_DIR = RAW_INCREMENTAL_DIR / "upcoming_games_results"

# ===== Ingested =====
INGESTED_GAMES_PATH = INGESTED_DIR / "historical" / "games"

# ===== Processed =====
TEAMS_CITIES_CONFERENCE_HISTORY_PROCESSED_PATH = (
    PROCESSED_DIR / "TeamsHistoriesConferenceNBALookUpTable.csv"
)
PROCESSED_LEAGUE_SCHEDULE_PATH = PROCESSED_DIR / "league_schedule.csv"
REGULAR_SEASON_GAMES_PATH = PROCESSED_DIR / "regular_season" / "games.csv"
NON_POSITIVE_SCORE_PATH = PROCESSED_DIR / "regular_season" / "non_positive_score.csv"
PLAYOFFS_GAMES_PATH = PROCESSED_DIR / "playoffs" / "games"

# Feature tables paths
REGULAR_SEASON_FEAURES_DIR = PROCESSED_DIR / "regular_season" / "features"
TEAMS_HOME_RECORDS_PATH = REGULAR_SEASON_FEAURES_DIR / "teams_home_record.csv"
TEAMS_AWAY_RECORDS_PATH = REGULAR_SEASON_FEAURES_DIR / "teams_away_record.csv"
TEAMS_RECORDS_PATH = REGULAR_SEASON_FEAURES_DIR / "teams_records.csv"
TEAMS_HOME_PTS_DIFF_PATH = REGULAR_SEASON_FEAURES_DIR / "teams_home_pts_diff.csv"
TEAMS_AWAY_PTS_DIFF_PATH = REGULAR_SEASON_FEAURES_DIR / "teams_away_pts_diff.csv"
TEAMS_PTS_DIFF_PATH = REGULAR_SEASON_FEAURES_DIR / "teams_pts_diff.csv"
EAST_WEST_RECORDS_PATH = REGULAR_SEASON_FEAURES_DIR / "east_west_record.csv"
EAST_WEST_RECORDS_AT_EAST_PATH = (
    REGULAR_SEASON_FEAURES_DIR / "east_west_record_at_east.csv"
)
EAST_WEST_RECORDS_AT_WEST_PATH = (
    REGULAR_SEASON_FEAURES_DIR / "east_west_record_at_west.csv"
)
RESTED_DAYS_PATH = REGULAR_SEASON_FEAURES_DIR / "rested_days.csv"
GAMES_FEATURES_PATH = REGULAR_SEASON_FEAURES_DIR / "games_features.csv"
