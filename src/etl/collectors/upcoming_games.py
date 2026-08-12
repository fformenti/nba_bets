"""Decide which scheduled games to collect next.

A game is *unresolved* when the schedule knows about it and the history table
does not. The next slate is the earliest date that still has unresolved games,
which finishes a partially-collected day before moving on and — unlike asking
only for dates after the frontier — can also reach a makeup game rescheduled
into a date the pipeline has already passed.

Games already parked as postponed or quarantined are excluded, so an unanswered
game cannot pin the slate to its date forever. Anything left unresolved further
than ``UNRESOLVED_GRACE_DAYS`` behind the frontier is quarantined here as a
second line of defence, for games that never made it into the pending directory
to have their attempts counted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config.constants import FETCH_ATTEMPTS_KEY, UNRESOLVED_GRACE_DAYS
from src.config.paths import (
    PROCESSED_LEAGUE_SCHEDULE_PATH,
    INGESTED_GAMES_UPDATED_HISTORY_PATH,
    POSTPONED_GAMES_DIR,
    UNRESOLVED_GAMES_DIR,
    UPCOMING_GAMES_DIR,
    TEAMS_CITIES_CONFERENCE_HISTORY_PROCESSED_PATH,
)
from src.etl.reference.teams_history import load_teams_history_table
from src.etl.transformation.add_conference import add_conference
from src.etl.utils.common import get_nba_season, add_neutral_court_game_flag
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)
setup_logging(level="INFO")


def _load_games_played(games_path: Path) -> pd.DataFrame:
    games = pd.read_csv(
        games_path,
        parse_dates=["gameDate"],
        low_memory=False,
    )
    if games.empty:
        raise ValueError(f"No games found in {games_path}")
    if games["gameDate"].dtype == object:
        games["gameDate"] = pd.to_datetime(games["gameDate"], errors="coerce")
    if games["gameDate"].isna().all():
        raise ValueError(f"All gameDate values are invalid in {games_path}")
    return games


def _load_schedule(schedule_path: Path) -> pd.DataFrame:
    schedule = pd.read_csv(schedule_path)
    if schedule.empty:
        # Guard the pruning in _save_upcoming_games: an empty schedule would make
        # every pending game look unwanted and silently clear the queue.
        raise ValueError(f"No games found in {schedule_path}")
    schedule = schedule.rename(
        columns={"homeTeamId": "hometeamId", "awayTeamId": "awayteamId"}
    )
    schedule["gameDateTimeEst"] = pd.to_datetime(
        schedule["gameDateTimeEst"], utc=True, errors="coerce"
    )
    schedule["gameDateTimeEstLocal"] = schedule["gameDateTimeEst"].dt.tz_convert(None)
    schedule["gameDate"] = schedule["gameDateTimeEstLocal"].dt.date
    return schedule


def _parked_game_ids(*directories: Path) -> set[int]:
    """gameIds already parked as postponed or quarantined.

    These are deliberately not collected again: they are waiting on a makeup
    date or on a human, and re-emitting them would put back the exact deadlock
    the quarantine exists to break.
    """
    parked: set[int] = set()
    for directory in directories:
        if directory is None or not directory.exists():
            continue
        for path in directory.glob("*.json"):
            try:
                parked.add(int(path.stem))
            except ValueError:
                logger.warning("Ignoring %s: filename is not a gameId", path.name)
    return parked


def _serialize_payload(payload: dict) -> str:
    """Render a payload the way this collector writes it.

    Goes through pandas so schedule rows keep working: they carry Timestamps and
    numpy scalars that ``json.dumps`` refuses.
    """
    return pd.Series(payload).to_json(date_format="iso")


def _write_payload_if_changed(payload: dict, path: Path) -> bool:
    """Write ``payload`` only when it differs from what is already on disk.

    Comparison is on the parsed content, not the bytes: a payload the results
    collector has touched was re-serialised with indentation, and rewriting it
    just to reflow whitespace would churn the file's mtime on every run.

    Returns whether the file was written.
    """
    serialized = _serialize_payload(payload)
    if path.exists():
        try:
            if json.loads(path.read_text(encoding="utf-8")) == json.loads(serialized):
                return False
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Unreadable payload — replacing it is the repair.
            logger.warning("Rewriting %s: existing payload is not valid JSON", path.name)
    path.write_text(serialized, encoding="utf-8")
    return True


def _quarantine_stragglers(
    stragglers: pd.DataFrame, unresolved_dir: Path, frontier: pd.Timestamp
) -> None:
    """Park scheduled games the pipeline has left too far behind.

    Writing a stub is enough: the file's only job is to keep the gameId out of
    the unresolved set. ``make retry-unresolved`` puts them back.
    """
    if stragglers.empty:
        return
    unresolved_dir.mkdir(parents=True, exist_ok=True)
    for _, row in stragglers.iterrows():
        game_id = int(row["gameId"])
        payload = row.to_dict()
        payload["_quarantineReason"] = (
            f"scheduled {row['gameDate']}, more than {UNRESOLVED_GRACE_DAYS} day(s) "
            f"behind the frontier {frontier.date()}"
        )
        if not _write_payload_if_changed(payload, unresolved_dir / f"{game_id}.json"):
            continue
        logger.warning(
            "Quarantined %s: scheduled %s but never collected (frontier %s)",
            game_id,
            row["gameDate"],
            frontier.date(),
        )


def _get_next_upcoming_games(
    schedule: pd.DataFrame,
    games_played: pd.DataFrame,
    postponed_dir: Path | None = None,
    unresolved_dir: Path | None = None,
    grace_days: int = UNRESOLVED_GRACE_DAYS,
) -> pd.DataFrame:
    """The earliest slate of scheduled games that history does not yet know."""
    frontier = games_played["gameDate"].max()
    logger.info(f"History frontier: {frontier.date()}")

    # A postponed row records a game that did not happen, so it does not settle
    # anything: the fixture is still owed. Counting those rows as played is what
    # would strand a makeup game forever — the schedule moves it to a new date
    # while history keeps insisting the original was already dealt with.
    settled = games_played
    if "postponed" in games_played.columns:
        settled = games_played[
            pd.to_numeric(games_played["postponed"], errors="coerce").fillna(0) != 1
        ]
        owed = len(games_played) - len(settled)
        if owed:
            logger.info(f"{owed} postponed row(s) in history still owe a fixture")

    played_ids = set(
        pd.to_numeric(settled["gameId"], errors="coerce").dropna().astype(int)
    )
    parked_ids = _parked_game_ids(postponed_dir, unresolved_dir)
    if parked_ids:
        logger.info(f"Skipping {len(parked_ids)} parked game(s)")

    schedule_ids = pd.to_numeric(schedule["gameId"], errors="coerce")
    unresolved = schedule[
        schedule_ids.notna()
        & ~schedule_ids.isin(played_ids)
        & ~schedule_ids.isin(parked_ids)
    ].copy()

    if unresolved.empty:
        logger.warning("Every scheduled game is already in history.")
        return unresolved

    # gameDate is a date object here; compare on timestamps to get a day delta.
    days_behind = (frontier.normalize() - pd.to_datetime(unresolved["gameDate"])).dt.days
    stragglers = unresolved[days_behind > grace_days]
    if not stragglers.empty and unresolved_dir is not None:
        _quarantine_stragglers(stragglers, unresolved_dir, frontier)
        unresolved = unresolved.drop(stragglers.index)

    if unresolved.empty:
        logger.warning("No collectable games remain after quarantining stragglers.")
        return unresolved

    next_date = unresolved["gameDate"].min()
    upcoming_games = unresolved[unresolved["gameDate"] == next_date]
    logger.info(f"Next date: {next_date}, {len(upcoming_games)} game(s)")
    return upcoming_games


def _save_upcoming_games(upcoming_games: pd.DataFrame, output_path: Path) -> None:
    """Write one payload per upcoming game, and drop payloads no longer wanted.

    Pruning matters: without it a game that moved on — collected, postponed,
    quarantined, or simply rescheduled out of this slate — would linger and be
    re-fetched forever.

    A game still in the slate has its payload refreshed from the schedule, so a
    changed tip-off time reaches history instead of the stale one the payload
    was first written with. Its attempt counter is carried across, and a payload
    that has not actually changed is not rewritten.
    """
    if output_path is None:
        return

    output_path.mkdir(parents=True, exist_ok=True)

    wanted: set[str] = set()
    refreshed = 0
    for _, row in upcoming_games.iterrows():
        game_id = row["gameId"]
        if pd.isna(game_id):
            continue
        json_path = output_path / f"{int(game_id)}.json"
        wanted.add(json_path.name)

        payload = row.to_dict()
        if json_path.exists():
            # The schedule is authoritative for everything about the fixture —
            # a rescheduled game must not keep writing its old date into history
            # — but the attempt counter is the collector's bookkeeping and would
            # be lost by a blind rewrite.
            try:
                existing = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                existing = {}
            if FETCH_ATTEMPTS_KEY in existing:
                payload[FETCH_ATTEMPTS_KEY] = existing[FETCH_ATTEMPTS_KEY]
            if _write_payload_if_changed(payload, json_path):
                refreshed += 1
                logger.info("Refreshed pending game %s from the schedule", json_path.name)
            continue

        _write_payload_if_changed(payload, json_path)

    for stale in output_path.glob("*.json"):
        if stale.name not in wanted:
            stale.unlink()
            logger.info("Removed stale pending game %s", stale.name)

    logger.info(
        "Saved %d upcoming game(s) to %s (%d refreshed)",
        len(wanted),
        output_path,
        refreshed,
    )


def get_upcoming_games(
    postponed_dir: Path = POSTPONED_GAMES_DIR,
    unresolved_dir: Path = UNRESOLVED_GAMES_DIR,
    output_dir: Path = UPCOMING_GAMES_DIR,
) -> None:
    schedule = _load_schedule(PROCESSED_LEAGUE_SCHEDULE_PATH)
    games_played = _load_games_played(INGESTED_GAMES_UPDATED_HISTORY_PATH)
    # To do - Create a script called process schedule and rename raw_games.py to process_raw_games.py
    schedule["gameDateOnlyStr"] = schedule["gameDate"].apply(
        lambda x: x.strftime("%Y-%m-%d")
    )
    upcoming_games = _get_next_upcoming_games(
        schedule,
        games_played,
        postponed_dir=postponed_dir,
        unresolved_dir=unresolved_dir,
    )

    if upcoming_games.empty:
        # Still prune: whatever is pending is no longer wanted.
        _save_upcoming_games(upcoming_games, output_dir)
        return

    upcoming_games["season"] = upcoming_games["gameDateTimeEstLocal"].apply(
        get_nba_season
    )
    teams_history = load_teams_history_table(
        TEAMS_CITIES_CONFERENCE_HISTORY_PROCESSED_PATH
    )
    upcoming_games = add_conference(
        upcoming_games, teams_history, ignore_winner_column=True
    )

    # Add neutral court game flag. gameLabel is kept rather than dropped: it and
    # gameSubLabel are the schedule's own description of what kind of game this
    # is, and the results payload has no other way to carry that to the append
    # step — which is what keeps NBA Cup and playoff games out of the
    # regular-season table.
    upcoming_games = add_neutral_court_game_flag(
        upcoming_games, game_label_column="gameLabel", drop_label_column=False
    )

    upcoming_games = upcoming_games.drop(columns=["gameDate"], errors="ignore")
    upcoming_games = upcoming_games.rename(columns={"gameDateTimeEstLocal": "gameDate"})
    upcoming_games = upcoming_games[
        [
            "gameId",
            "gameDate",
            "gameDateOnlyStr",
            "arenaCity",
            "arenaName",
            "hometeamId",
            "homeTeamCity",
            "homeTeamName",
            "hometeamConference",
            "awayteamId",
            "awayTeamCity",
            "awayTeamName",
            "awayteamConference",
            "winnerteamConference",
            "season",
            "is_neutral_court_game",
            "gameLabel",
            "gameSubLabel",
        ]
    ].copy()

    _save_upcoming_games(upcoming_games, output_dir)
    return
