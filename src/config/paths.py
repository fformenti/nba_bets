"""
Path constants for data files and directories.

All paths are relative to PROJECT_ROOT for portability.
"""

from pathlib import Path

# Project root directory
script_dir = Path(__file__).parent
PROJECT_ROOT = script_dir.parent.parent

# Filenames
LOCAL_TEAMS_HISTORY_CITIES_CONFERENCES_FILENAME = "teams_history_expanded.csv"

# Raw data paths
LOCAL_RAW_GAMES_PATH = (
    PROJECT_ROOT / "data" / "raw" / "historical" / "games" / "Games.csv"
)
LOCAL_RAW_INCREMENTAL_DIR = PROJECT_ROOT / "data" / "raw" / "incremental"
LOCAL_RAW_INCREMENTAL_ARCHIVE_DIR = LOCAL_RAW_INCREMENTAL_DIR / "archive"
LOCAL_LEAGUE_SCHEDULE_PATH = (
    PROJECT_ROOT / "data" / "raw" / "historical" / "LeagueSchedule24_25.csv"
)

# Ingested data paths
LOCAL_INGESTED_GAMES_PATH = PROJECT_ROOT / "data" / "ingested" / "historical" / "games"
LOCAL_INGESTED_INCREMENTAL_GAMES_PATH = (
    PROJECT_ROOT / "data" / "ingested" / "incremental" / "games"
)

# Processed data paths
LOCAL_PROCESSED_FOLDER = PROJECT_ROOT / "data" / "processed"
LOCAL_REGULAR_SEASON_GAMES_PATH = (
    LOCAL_PROCESSED_FOLDER / "regular_season" / "games.csv"
)
LOCAL_PLAYOFFS_GAMES_PATH = LOCAL_PROCESSED_FOLDER / "playoffs" / "games"

# Feature tables paths
LOCAL_TEAMS_HISTORY_CITIES_CONFERENCES_PATH = (
    LOCAL_PROCESSED_FOLDER / LOCAL_TEAMS_HISTORY_CITIES_CONFERENCES_FILENAME
)
LOCAL_TEAMS_HOME_RECORDS_PATH = LOCAL_PROCESSED_FOLDER / "teams_home_record.csv"
LOCAL_TEAMS_AWAY_RECORDS_PATH = LOCAL_PROCESSED_FOLDER / "teams_away_record.csv"
LOCAL_TEAMS_RECORDS_PATH = LOCAL_PROCESSED_FOLDER / "teams_records.csv"
LOCAL_TEAMS_HOME_PTS_DIFF_PATH = LOCAL_PROCESSED_FOLDER / "teams_home_pts_diff.csv"
LOCAL_TEAMS_AWAY_PTS_DIFF_PATH = LOCAL_PROCESSED_FOLDER / "teams_away_pts_diff.csv"
LOCAL_TEAMS_PTS_DIFF_PATH = LOCAL_PROCESSED_FOLDER / "teams_pts_diff.csv"
LOCAL_EAST_WEST_RECORDS_PATH = LOCAL_PROCESSED_FOLDER / "east_west_record.csv"
LOCAL_EAST_WEST_RECORDS_AT_EAST_PATH = (
    LOCAL_PROCESSED_FOLDER / "east_west_record_at_east.csv"
)
LOCAL_EAST_WEST_RECORDS_AT_WEST_PATH = (
    LOCAL_PROCESSED_FOLDER / "east_west_record_at_west.csv"
)
LOCAL_RESTED_DAYS_PATH = LOCAL_PROCESSED_FOLDER / "rested_days.csv"
LOCAL_GAMES_FEATURES_PATH = LOCAL_PROCESSED_FOLDER / "games_features.csv"
