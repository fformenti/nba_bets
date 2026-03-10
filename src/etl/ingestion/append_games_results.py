"""Add game results from upcoming games results to historical raw games."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.paths import (
    INGESTED_GAMES_PATH,
    INGESTED_GAMES_UPDATED_HISTORY_PATH,
    UPCOMING_GAMES_RESULTS_DIR,
)
from src.etl.utils.common import coerce_numeric_columns
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


EXPECTED_COLUMNS = [
    "gameId",
    "gameDate",
    "gameDateOnlyStr",
    "season",
    "hometeamCity",
    "hometeamName",
    "hometeamId",
    "awayteamCity",
    "awayteamName",
    "awayteamId",
    "homeScore",
    "awayScore",
    "winner",
    "overtimes",
    "postponed",
    "gameType",
    "attendance",
    "arenaId",
    "gameLabel",
    "gameSubLabel",
    "seriesGameNumber",
]


def _read_json_file(file_path: Path) -> dict[str, Any]:
    """Read a JSON file and return its contents."""
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _transform_game_result_to_dataframe_row(
    game_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Transform a single game result JSON to a dataframe row matching historical games schema.

    Args:
        game_data: Dictionary containing game result data from JSON file

    Returns:
        Dictionary with keys matching historical games column names
    """
    # Extract scores
    home_score = game_data.get("homeTeamFinalScore")
    away_score = game_data.get("awayTeamFinalScore")
    hometeam_id = game_data.get("hometeamId")
    awayteam_id = game_data.get("awayteamId")
    winner = game_data.get("winner")
    postponed = game_data.get("postponed")
    overtimes = game_data.get("overtimes")
    # Parse gameDate from ISO format
    game_date_str = game_data.get("gameDateOnlyStr")
    game_date_timestamp = game_data.get("gameDate")
    if game_date_str:
        game_date = pd.to_datetime(game_date_timestamp)
    else:
        game_date = None

    # Map columns from JSON to historical games schema
    row = {
        "gameId": game_data.get("gameId"),
        "gameDate": game_date,
        "gameDateOnlyStr": game_date_str,
        "season": game_data.get("season"),
        "hometeamCity": game_data.get("homeTeamCity", ""),
        "hometeamName": game_data.get("homeTeamName", ""),
        "hometeamId": hometeam_id,
        "awayteamCity": game_data.get("awayTeamCity", ""),
        "awayteamName": game_data.get("awayTeamName", ""),
        "awayteamId": awayteam_id,
        "homeScore": home_score,
        "awayScore": away_score,
        "winner": winner,
        "overtimes": overtimes,
        "postponed": postponed,
        "gameType": "",  # Empty string as default
        "attendance": game_data.get("attendance"),
        "arenaId": None,  # Not present in JSON
        "gameLabel": None,
        "gameSubLabel": None,
        "seriesGameNumber": None,
    }

    return row


def load_upcoming_game_results(
    results_dir: Path,
) -> tuple[pd.DataFrame, dict[int, Path]]:
    """
    Load all JSON files from upcoming games results directory and transform to dataframe.

    Args:
        results_dir: Path to directory containing JSON game result files

    Returns:
        Tuple of (DataFrame with columns matching historical games schema,
                 dictionary mapping gameId to JSON file path)
    """
    if not results_dir.exists():
        logger.warning(f"Results directory does not exist: {results_dir}")
        return pd.DataFrame(), {}

    # Get JSON files, excluding those in the archive subdirectory
    json_files = [f for f in sorted(results_dir.glob("*.json"))]
    if not json_files:
        logger.warning(f"No JSON files found in {results_dir}")
        return pd.DataFrame(), {}

    logger.info(f"Found {len(json_files)} JSON files to process")

    rows = []
    for json_file in json_files:
        try:
            game_data = _read_json_file(json_file)
            row = _transform_game_result_to_dataframe_row(game_data)
            rows.append(row)
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
            continue

    if not rows:
        logger.warning("No valid game results found")
        return pd.DataFrame(), {}

    df = pd.DataFrame(rows)

    # Ensure data types match historical games
    coerce_numeric_columns(df, ["gameId", "hometeamId", "awayteamId"])
    coerce_numeric_columns(df, ["winner", "arenaId"], dtype="Int64")
    df["attendance"] = pd.to_numeric(df["attendance"], errors="coerce")

    # Ensure gameDate is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["gameDate"]):
        df["gameDate"] = pd.to_datetime(df["gameDate"], errors="coerce")

    logger.info(f"Successfully loaded {len(df)} game results")
    return df


def load_historical_games(historical_games_path: Path) -> pd.DataFrame:
    """
    Load historical games from CSV file.

    Args:
        historical_games_path: Path to historical games CSV file

    Returns:
        DataFrame with historical games
    """
    if not historical_games_path.exists():
        logger.warning(f"Historical games file does not exist: {historical_games_path}")
        return pd.DataFrame()

    logger.info(f"Loading historical games from {historical_games_path}")
    df = pd.read_csv(historical_games_path, parse_dates=["gameDate"], low_memory=False)
    logger.info(f"Loaded {len(df)} historical games")
    return df


def add_game_results_to_historical(
    upcoming_results_dir: Path | None = None,
    historical_games_path: Path | None = None,
    output_path: Path | None = None,
    keep_old_ids: bool = True,
) -> pd.DataFrame:
    """
    Add game results from upcoming games results to historical raw games.

    Args:
        upcoming_results_dir: Path to directory containing upcoming game results JSON files.
                             Defaults to UPCOMING_GAMES_RESULTS_DIR.
        historical_games_path: Path to historical games CSV file.
                               Defaults to RAW_GAMES_PATH.
        output_path: Path where to save the updated historical games.
                     Defaults to RAW_GAMES_UPDATED_HISTORY_PATH

    Returns:
        DataFrame with combined historical and new game results
    """
    # Set default paths
    if upcoming_results_dir is None:
        upcoming_results_dir = UPCOMING_GAMES_RESULTS_DIR
    if historical_games_path is None:
        historical_games_path = INGESTED_GAMES_PATH
    if output_path is None:
        output_path = INGESTED_GAMES_UPDATED_HISTORY_PATH

    # Load historical games
    historical_df = load_historical_games(historical_games_path)

    # Load upcoming game results and track file mappings
    upcoming_df = load_upcoming_game_results(upcoming_results_dir)
    # games_ids = upcoming_df["gameId"].tolist()
    if upcoming_df.empty:
        logger.warning("No upcoming game results to add")
        return historical_df

    # Ensure both dataframes have the same columns in the same order
    historical_df = historical_df.reindex(columns=EXPECTED_COLUMNS)
    upcoming_df = upcoming_df.reindex(columns=EXPECTED_COLUMNS)

    historical_df = historical_df[~historical_df["gameId"].isin(upcoming_df["gameId"])]

    # _archive_processed_games(archive_ids, game_id_to_file, archive_dir)
    combined_df = pd.concat([historical_df, upcoming_df], ignore_index=True)
    # Sort by gameDate and gameId for consistency
    combined_df = combined_df.sort_values(
        ["gameDate", "gameId"], na_position="last"
    ).reset_index(drop=True)

    # Save to output path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(combined_df)} games to {output_path}")
    logger.info(f"Added {len(upcoming_df)} new games to historical data")

    return combined_df


def main() -> None:
    """Main entry point for the script."""
    setup_logging(level="INFO")
    keep_old_ids = False
    add_game_results_to_historical(keep_old_ids=keep_old_ids)


if __name__ == "__main__":
    main()
