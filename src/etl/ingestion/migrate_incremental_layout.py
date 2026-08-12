"""One-time move from the rebuild-everything layout to the staged one.

Historically ``append-game-results`` rebuilt the whole history on every run from
the seed CSV plus *every* JSON ever written to the results inbox. That made the
inbox load-bearing: those files were the only copy of the games collected since
the seed was frozen, so archiving them would have silently dropped a month of
basketball.

This module promotes the rebuilt table to the authoritative one so the inbox can
finally be drained. It is idempotent — once the inbox is empty and the history is
clean it does nothing — and it refuses to move a single file until it has proven
every game in the inbox is already recorded in history.

Run once, via ``make migrate-incremental``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from src.config.constants import UNRESOLVED_GRACE_DAYS
from src.config.paths import (
    ARCHIVE_RESULTS_DIR,
    INGESTED_GAMES_UPDATED_HISTORY_PATH,
    MANUAL_RESULTS_DIR,
    POSTPONED_GAMES_DIR,
    UNRESOLVED_GAMES_DIR,
    UPCOMING_GAMES_DIR,
    UPCOMING_GAMES_RESULTS_DIR,
)
from src.etl.utils.common import read_json
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Directories the staged flow expects to exist, even when empty.
_STAGED_DIRS = (
    UPCOMING_GAMES_DIR,
    UPCOMING_GAMES_RESULTS_DIR,
    ARCHIVE_RESULTS_DIR,
    POSTPONED_GAMES_DIR,
    UNRESOLVED_GAMES_DIR,
    MANUAL_RESULTS_DIR,
)


class MigrationAborted(RuntimeError):
    """Raised when migrating would lose games. Nothing has been moved."""


def _game_id(payload: dict) -> int | None:
    raw = payload.get("gameId")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def ensure_staged_dirs() -> None:
    """Create every directory the staged flow expects, with a .gitkeep."""
    for directory in _STAGED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        keep = directory / ".gitkeep"
        if not keep.exists():
            keep.touch()


def find_unrecorded_games(
    results_dir: Path, history: pd.DataFrame
) -> dict[int, Path]:
    """Inbox files whose game is missing from ``history``, keyed by gameId.

    A non-empty result means the inbox still holds the only copy of those games
    and must not be archived.
    """
    recorded = set(pd.to_numeric(history["gameId"], errors="coerce").dropna().astype(int))
    unrecorded: dict[int, Path] = {}
    for path in sorted(results_dir.glob("*.json")):
        game_id = _game_id(read_json(path))
        if game_id is None:
            logger.warning("Skipping %s: unreadable gameId", path.name)
            continue
        if game_id not in recorded:
            unrecorded[game_id] = path
    return unrecorded


def clean_history(history: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list]]:
    """Drop rows that cannot describe a real game, and collapse duplicate ids.

    Two defects the old rebuild left behind:

    * Rows with no team ids. All-Star weekend events (Rising Stars, the skills
      contests) reach the schedule with placeholder teams and were appended as
      postponed games.
    * The same gameId twice, because the dedupe key was
      ``(date, home, away)`` and a rescheduled game changes date. The played row
      is the true one; the postponed row is a ghost of the original fixture.
    """
    dropped: dict[str, list] = {"no_teams": [], "duplicate": []}

    ids = pd.to_numeric(history["gameId"], errors="coerce")
    home = pd.to_numeric(history["hometeamId"], errors="coerce")
    away = pd.to_numeric(history["awayteamId"], errors="coerce")

    teamless = home.isna() | away.isna() | (home == 0) | (away == 0)
    if teamless.any():
        dropped["no_teams"] = sorted(ids[teamless].dropna().astype(int).tolist())
        history = history.loc[~teamless].copy()
        ids = ids.loc[~teamless]

    # Among rows sharing a gameId, a played row beats a postponed one; ties fall
    # back to the later date, which is the fixture that actually took place.
    duplicated_ids = ids[ids.duplicated(keep=False)].dropna().astype(int).unique()
    if len(duplicated_ids):
        dropped["duplicate"] = sorted(duplicated_ids.tolist())
        history = history.assign(_gid=ids, _pp=pd.to_numeric(history["postponed"], errors="coerce"))
        history = (
            history.sort_values(["_gid", "_pp", "gameDate"], ascending=[True, True, True])
            .drop_duplicates(subset="_gid", keep="first")
            .drop(columns=["_gid", "_pp"])
        )

    return history, dropped


def quarantine_abandoned_pending(
    pending_dir: Path, unresolved_dir: Path, frontier: pd.Timestamp
) -> list[Path]:
    """Park pending games the pipeline has already moved past.

    A game still sitting in ``upcoming_games/`` whose scheduled date is strictly
    behind the history frontier was never answered by the results source. Left
    there it blocks the selector forever, because the frontier cannot advance
    past a date that still has unresolved games.

    Games scheduled *on* the frontier are left alone. The selector deliberately
    finishes a partially-collected day before moving on, so the current slate
    normally sits on the frontier date — quarantining it would mean a second run
    of this migration parked a perfectly healthy set of pending games.
    """
    moved: list[Path] = []
    unresolved_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(pending_dir.glob("*.json")):
        payload = read_json(path)
        game_date = pd.to_datetime(payload.get("gameDate"), errors="coerce")
        if pd.isna(game_date) or game_date.normalize() >= frontier.normalize():
            continue
        destination = unresolved_dir / path.name
        shutil.move(str(path), str(destination))
        moved.append(destination)
        logger.warning(
            "Quarantined %s: scheduled %s, never resolved (frontier %s)",
            path.name,
            game_date.date(),
            frontier.date(),
        )
    return moved


def migrate_incremental_layout(
    results_dir: Path | None = None,
    pending_dir: Path | None = None,
    archive_dir: Path | None = None,
    unresolved_dir: Path | None = None,
    history_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Promote the rebuilt history and drain the inbox. Safe to re-run."""
    results_dir = results_dir or UPCOMING_GAMES_RESULTS_DIR
    pending_dir = pending_dir or UPCOMING_GAMES_DIR
    archive_dir = archive_dir or ARCHIVE_RESULTS_DIR
    unresolved_dir = unresolved_dir or UNRESOLVED_GAMES_DIR
    history_path = history_path or INGESTED_GAMES_UPDATED_HISTORY_PATH

    if not history_path.exists():
        raise MigrationAborted(
            f"No history table at {history_path}. Run `make append-game-results` first — "
            "migrating without it would archive the only copy of the collected games."
        )

    ensure_staged_dirs()

    history = pd.read_csv(history_path, parse_dates=["gameDate"], low_memory=False)
    rows_before = len(history)

    unrecorded = find_unrecorded_games(results_dir, history)
    if unrecorded:
        raise MigrationAborted(
            f"{len(unrecorded)} game(s) in {results_dir.name} are not in the history table: "
            f"{sorted(unrecorded)[:10]}{' …' if len(unrecorded) > 10 else ''}. "
            "Run `make append-game-results` first. Nothing was moved."
        )

    history, dropped = clean_history(history)
    for game_id in dropped["no_teams"]:
        logger.warning("Dropping %s from history: no team ids — not a real game", game_id)
    for game_id in dropped["duplicate"]:
        logger.warning("Collapsing duplicate rows for %s, keeping the played fixture", game_id)

    frontier = history["gameDate"].max()
    to_archive = sorted(results_dir.glob("*.json"))

    if dry_run:
        logger.info(
            "[dry run] would archive %d file(s), drop %d row(s), frontier %s",
            len(to_archive),
            rows_before - len(history),
            frontier.date(),
        )
        return {
            "archived": len(to_archive),
            "rows_dropped": rows_before - len(history),
            "quarantined": 0,
        }

    if rows_before != len(history):
        history.to_csv(history_path, index=False)
        logger.info("Rewrote %s: %d → %d rows", history_path.name, rows_before, len(history))

    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in to_archive:
        shutil.move(str(path), str(archive_dir / path.name))
    if to_archive:
        logger.info("Archived %d consumed result(s) to %s", len(to_archive), archive_dir)

    quarantined = quarantine_abandoned_pending(pending_dir, unresolved_dir, frontier)

    remaining = pd.to_numeric(history["gameId"], errors="coerce").dropna()
    if remaining.duplicated().any():
        raise MigrationAborted("gameId is still not unique after migration — investigate.")

    logger.info(
        "Migration complete: %d archived, %d row(s) dropped, %d quarantined, frontier %s",
        len(to_archive),
        rows_before - len(history),
        len(quarantined),
        frontier.date(),
    )
    logger.info(
        "Grace window for future stragglers: %d day(s) behind the frontier.",
        UNRESOLVED_GRACE_DAYS,
    )
    return {
        "archived": len(to_archive),
        "rows_dropped": rows_before - len(history),
        "quarantined": len(quarantined),
    }
