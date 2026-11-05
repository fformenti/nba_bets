"""Watch parked games for a makeup date, and release them when one appears.

A postponed game is not written to history — there is no result to record — so
without something watching it, it would simply vanish from the pipeline. Parking
it keeps the selector from re-emitting it every run; this module is what lets it
back in once the league reschedules it.

Releasing a game means deleting its parked file. The selector then sees a
scheduled game that history does not know about and collects it on its new date,
which is the same path any other game takes. Nothing needs to re-queue it by
hand.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.paths import (
    POSTPONED_GAMES_DIR,
    PROCESSED_LEAGUE_SCHEDULE_PATH,
    UNRESOLVED_GAMES_DIR,
)
from src.etl.utils.common import read_json
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _scheduled_tipoffs(schedule_path: Path) -> dict[int, pd.Timestamp]:
    """gameId → scheduled tip-off, as naive Eastern wall time.

    The schedule stamps a "Z" on times that are Eastern, so the label is
    dropped rather than converted — the same convention as the selector's
    ``_load_schedule``, which is what wrote the parked payload's ``gameDate``.
    """
    schedule = pd.read_csv(schedule_path)
    tipoffs = pd.to_datetime(schedule["gameDateTimeEst"], utc=True, errors="coerce")
    tipoffs = tipoffs.dt.tz_convert(None)
    ids = pd.to_numeric(schedule["gameId"], errors="coerce")
    return {
        int(game_id): tipoff
        for game_id, tipoff in zip(ids, tipoffs, strict=True)
        if pd.notna(game_id) and pd.notna(tipoff)
    }


def _parked_tipoff(payload: dict) -> pd.Timestamp | None:
    tipoff = pd.to_datetime(payload.get("gameDate"), errors="coerce")
    if pd.isna(tipoff):
        tipoff = pd.to_datetime(payload.get("gameDateOnlyStr"), errors="coerce")
    if pd.isna(tipoff):
        return None
    return tipoff.tz_localize(None) if tipoff.tzinfo is not None else tipoff


def reconcile_postponed_games(
    postponed_dir: Path | None = None,
    schedule_path: Path | None = None,
) -> list[int]:
    """Release parked games whose scheduled tip-off has moved. Returns their ids.

    The whole tip-off is compared, not just the calendar date: a game suspended
    and resumed later the same day, or pushed back a few hours, would otherwise
    never be released and its result never collected.
    """
    postponed_dir = postponed_dir or POSTPONED_GAMES_DIR
    schedule_path = schedule_path or PROCESSED_LEAGUE_SCHEDULE_PATH

    if not postponed_dir.exists():
        logger.info("No postponed directory yet; nothing to reconcile")
        return []

    parked = sorted(postponed_dir.glob("*.json"))
    if not parked:
        logger.info("No postponed games being watched")
        return []

    scheduled = _scheduled_tipoffs(schedule_path)
    released: list[int] = []

    for path in parked:
        payload = read_json(path)
        try:
            game_id = int(payload["gameId"])
        except (KeyError, TypeError, ValueError):
            logger.warning("Skipping %s: unreadable gameId", path.name)
            continue

        parked_tipoff = _parked_tipoff(payload)
        new_tipoff = scheduled.get(game_id)

        if new_tipoff is None:
            # Either the season's schedule has rolled over or the league dropped
            # the game. A cancellation is reported by the results source, which
            # moves the game to cancelled/; until then a human has to look.
            logger.warning(
                "Game %s is no longer on the schedule; leaving it parked. "
                "If it was cancelled, move %s to the cancelled directory.",
                game_id,
                path.name,
            )
            continue

        if new_tipoff == parked_tipoff:
            logger.info(
                "Game %s still shows %s; no makeup date yet", game_id, parked_tipoff
            )
            continue

        path.unlink()
        released.append(game_id)
        logger.info(
            "Game %s rescheduled %s → %s; released for collection",
            game_id,
            parked_tipoff,
            new_tipoff,
        )

    logger.info("Released %d of %d watched game(s)", len(released), len(parked))
    return released


def retry_unresolved_games(unresolved_dir: Path | None = None) -> list[int]:
    """Let quarantined games back into the pipeline, attempts reset.

    Dropping the quarantine file is the whole operation: the selector rebuilds
    the payload from the schedule on its next run, so a released game starts
    from a clean slate rather than inheriting the attempts that parked it.
    """
    unresolved_dir = unresolved_dir or UNRESOLVED_GAMES_DIR

    if not unresolved_dir.exists():
        logger.info("No quarantine directory yet; nothing to retry")
        return []

    quarantined = sorted(unresolved_dir.glob("*.json"))
    released: list[int] = []
    for path in quarantined:
        try:
            game_id = int(path.stem)
        except ValueError:
            logger.warning("Skipping %s: filename is not a gameId", path.name)
            continue
        path.unlink()
        released.append(game_id)
        logger.info("Released %s from quarantine", game_id)

    if released:
        logger.info(
            "Released %d game(s). Run `make fetch-upcoming-games` to re-queue them.",
            len(released),
        )
    else:
        logger.info("Nothing in quarantine")
    return released
