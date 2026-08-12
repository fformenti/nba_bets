"""Fold fetched game results into the history table.

This is a true append: the history table is read, the results inbox is added to
it, and each consumed file is moved to the archive. It used to rebuild the whole
table from every JSON ever written, which made the inbox load-bearing — the files
could never be cleared because they were the only record of everything collected.

Seeding the table from the raw historical archive is `ingest_raw_games`'s job, not
this module's.
"""

from __future__ import annotations
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.paths import (
    ARCHIVE_RESULTS_DIR,
    INGESTED_GAMES_UPDATED_HISTORY_PATH,
    UPCOMING_GAMES_RESULTS_DIR,
)
from src.etl.utils.common import (
    CANONICAL_INGESTED_COLUMNS,
    assert_unique_game_ids,
    atomic_write_csv,
    coerce_numeric_columns,
    deduplicate_games,
    enrich_games_locations,
    game_type_from_id,
    read_json,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


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

    # The gameId is the only reliable source of gameType for a collected game:
    # results payloads carry no such field, and the league schedule leaves
    # gameLabel blank for every playoff round. Without this every incrementally
    # collected game would arrive as an empty string, pass every filter in
    # filter_regular_season_games, and land in the regular-season table.
    game_id = game_data.get("gameId")
    game_type = game_type_from_id(game_id)
    if game_type is None:
        logger.warning("Cannot read a game type from gameId %r", game_id)

    # Map columns from JSON to historical games schema
    row = {
        "gameId": game_id,
        "gameDate": game_date,
        "gameDateOnlyStr": game_date_str,
        "season": game_data.get("season"),
        "hometeamPrename": game_data.get("homeTeamCity", ""),
        "hometeamName": game_data.get("homeTeamName", ""),
        "hometeamId": hometeam_id,
        "awayteamPrename": game_data.get("awayTeamCity", ""),
        "awayteamName": game_data.get("awayTeamName", ""),
        "awayteamId": awayteam_id,
        "homeScore": home_score,
        "awayScore": away_score,
        "winner": winner,
        "overtimes": overtimes,
        "postponed": postponed,
        "gameType": game_type if game_type is not None else "",
        "attendance": game_data.get("attendance"),
        "gameLabel": game_data.get("gameLabel"),
        "gameSubLabel": game_data.get("gameSubLabel"),
        # arenaId and seriesGameNumber are not present in the JSON — omitted here
        # (rather than set to None) so they don't become all-NA columns that trip
        # pandas' concat FutureWarning; the later reindex to
        # CANONICAL_INGESTED_COLUMNS fills them back in as NaN.
    }

    return row


def load_upcoming_game_results(
    results_dir: Path,
) -> tuple[pd.DataFrame, list[Path]]:
    """
    Load every result JSON in the inbox and transform it to the ingested schema.

    Args:
        results_dir: Path to directory containing JSON game result files

    Returns:
        Tuple of (DataFrame with columns matching historical games schema,
                 the files that produced it — the ones safe to archive)
    """
    if not results_dir.exists():
        logger.warning(f"Results directory does not exist: {results_dir}")
        return pd.DataFrame(), []

    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        logger.info(f"No new game results in {results_dir}")
        return pd.DataFrame(), []

    logger.info(f"Found {len(json_files)} JSON files to process")

    rows = []
    consumed: list[Path] = []
    for json_file in json_files:
        try:
            game_data = read_json(json_file)
            if not _is_played(game_data):
                # Belt and braces: only played games reach this directory, but a
                # postponed row here would become a permanent ghost fixture.
                logger.warning(
                    "Skipping %s: not a played game (status=%r, postponed=%r)",
                    json_file.name,
                    game_data.get("status"),
                    game_data.get("postponed"),
                )
                continue
            rows.append(_transform_game_result_to_dataframe_row(game_data))
            consumed.append(json_file)
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
            continue

    if not rows:
        logger.warning("No valid game results found")
        return pd.DataFrame(), []

    df = pd.DataFrame(rows)

    # Ensure data types match historical games
    coerce_numeric_columns(df, ["gameId", "hometeamId", "awayteamId"])
    coerce_numeric_columns(df, ["winner", "arenaId"], dtype="Int64")
    df["attendance"] = pd.to_numeric(df["attendance"], errors="coerce")

    # Ensure gameDate is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["gameDate"]):
        df["gameDate"] = pd.to_datetime(df["gameDate"], errors="coerce")

    logger.info(f"Successfully loaded {len(df)} game results")
    return df, consumed


def _is_played(game_data: dict[str, Any]) -> bool:
    """Whether this payload describes a game that actually happened.

    Trusts an explicit status when present and falls back to the ``postponed``
    flag for payloads written before statuses existed.
    """
    status = game_data.get("status")
    if status is not None:
        return str(status).lower() == "final"
    return not game_data.get("postponed")


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
    output_path: Path | None = None,
    archive_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Append fetched game results to the history table and archive what was used.

    Args:
        upcoming_results_dir: Directory holding the result JSON inbox.
                              Defaults to UPCOMING_GAMES_RESULTS_DIR.
        output_path: The history table, both read and written.
                     Defaults to INGESTED_GAMES_UPDATED_HISTORY_PATH.
        archive_dir: Where consumed result files are moved.
                     Defaults to ARCHIVE_RESULTS_DIR.

    Returns:
        DataFrame with the updated history
    """
    # Set default paths
    if upcoming_results_dir is None:
        upcoming_results_dir = UPCOMING_GAMES_RESULTS_DIR
    if output_path is None:
        output_path = INGESTED_GAMES_UPDATED_HISTORY_PATH
    if archive_dir is None:
        archive_dir = ARCHIVE_RESULTS_DIR

    historical_df = load_historical_games(output_path)

    upcoming_df, consumed = load_upcoming_game_results(upcoming_results_dir)
    if upcoming_df.empty:
        logger.info("No new game results to add — history left unchanged")
        return historical_df

    # Enrich upcoming results with location columns
    upcoming_df = enrich_games_locations(upcoming_df)

    # Drop any rows the new results supersede, matched by gameId
    historical_df = deduplicate_games(historical_df, upcoming_df)

    # Combine and align to canonical schema — filter out empty frames to avoid FutureWarning
    frames = [df for df in [historical_df, upcoming_df] if not df.empty]
    combined_df = pd.concat(frames, ignore_index=True)
    combined_df = combined_df.reindex(columns=CANONICAL_INGESTED_COLUMNS)
    # Sort by gameDate and gameId for consistency
    combined_df = combined_df.sort_values(
        ["gameDate", "gameId"], na_position="last"
    ).reset_index(drop=True)

    assert_unique_game_ids(combined_df, "append")

    # Atomic: this is the authoritative history table, and nothing downstream can
    # tell a truncated table from a genuinely short one.
    atomic_write_csv(combined_df, output_path)
    logger.info(f"Saved {len(combined_df)} games to {output_path}")
    logger.info(f"Added {len(upcoming_df)} new game(s) to the history table")

    # Only archive once the table is safely on disk: these files are the sole
    # record of the games until the write above succeeds.
    _archive_consumed(consumed, archive_dir)

    return combined_df


def _archive_consumed(consumed: list[Path], archive_dir: Path) -> None:
    """Move result files that are now recorded in history out of the inbox."""
    if not consumed:
        return
    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in consumed:
        shutil.move(str(path), str(archive_dir / path.name))
    logger.info(f"Archived {len(consumed)} consumed result(s) to {archive_dir}")
