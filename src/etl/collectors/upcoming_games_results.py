"""Ask a results source what happened to each pending game, and file it.

Where the outcome comes from is decided by the caller: this module takes any
:class:`~src.etl.collectors.results.base.ResultsSource` and knows nothing about
nba_api, scraping, or hand-entered files.

Every pending game ends up in exactly one place, decided by the status the
source reports and never by the scoreline:

``FINAL``
    Enriched with the score and moved to the results inbox, ready for
    ``append-game-results`` to fold into history.
``POSTPONED``
    Moved to the postponed directory, where ``reconcile-postponed`` watches for
    a makeup date. It is never written to history as a played game.
``SCHEDULED`` / ``IN_PROGRESS`` / ``UNKNOWN``
    Left pending. Once a game has been asked about ``MAX_FETCH_ATTEMPTS`` times
    *after its scheduled tip-off* without a terminal answer it is moved to the
    quarantine directory — otherwise a single game the source will never answer
    holds up the entire pipeline, because the history frontier cannot advance
    past a date that still has unresolved games.

Attempts are only counted once the game has tipped off. A source saying
``SCHEDULED`` about a game that has not started yet is a correct answer, not a
failure, so it must not burn an attempt: counting those would let a handful of
re-runs on the morning of a slate quarantine the whole evening's games.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pandas as pd

from src.config.constants import (
    FETCH_ATTEMPTS_KEY,
    MAX_FETCH_ATTEMPTS,
    SCHEDULE_TIMEZONE,
)
from src.config.paths import PROJECT_ROOT
from src.etl.collectors.results import DEFAULT_SOURCE, get_results_source
from src.etl.collectors.results.base import GameStatus, ResultsSource
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# stats.nba.com rate-limits aggressively; pace batch requests.
FETCH_DELAY_SECONDS = 4

# Re-exported so callers that reason about a pending payload's bookkeeping can
# keep importing it from the collector that owns the semantics.
ATTEMPTS_KEY = FETCH_ATTEMPTS_KEY


def _display_path(path: Path) -> str:
    """Project-relative path for logging, falling back to the absolute one.

    ``--output-dir`` may point anywhere, and a log line must never be the thing
    that fails a run.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _write_payload(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def _has_tipped_off(payload: dict) -> bool:
    """Whether the game's scheduled start has passed.

    ``gameDate`` carries the tip-off as Eastern wall time with no offset — the
    league schedule labels the column ``gameDateTimeEst`` and stamps a ``Z`` on
    values that are not actually UTC, and the collector keeps that convention
    naive (see ``upcoming_games._load_schedule``). So the comparison has to be
    against Eastern now, not UTC now.

    A payload with no usable ``gameDate`` counts as tipped off: an attempt
    counter that never advances would leave it pending forever.
    """
    tipoff = pd.to_datetime(payload.get("gameDate"), errors="coerce")
    if pd.isna(tipoff):
        return True
    if tipoff.tzinfo is not None:
        # The schedule stamps a "Z" on values that are not UTC. Strip the label
        # rather than converting, so the number keeps meaning Eastern wall time.
        tipoff = tipoff.tz_localize(None)
    now_et = pd.Timestamp.now(tz=SCHEDULE_TIMEZONE).tz_localize(None)
    return tipoff <= now_et


def resolve_winner_team_id(payload: dict, result) -> int:
    """Return the winning team's id, deriving it from scores when needed.

    Throughout the project ``winner`` is the winning *teamId*, not a flag. A
    source that only knows the scoreline (the placeholder, or any future feed
    that reports scores alone) leaves ``winner`` at 0; the team ids live on the
    game payload, so the mapping is applied here, once, for every source.
    """
    if result.winner:
        return result.winner
    if result.home_score == result.away_score:
        return 0
    is_home_win = result.home_score > result.away_score
    return payload["hometeamId"] if is_home_win else payload["awayteamId"]


def enrich_upcoming_game_result(
    input_path: Path,
    output_dir: Path,
    source: ResultsSource,
    postponed_dir: Path | None = None,
    unresolved_dir: Path | None = None,
    max_attempts: int = MAX_FETCH_ATTEMPTS,
) -> tuple[GameStatus, Path | None]:
    """Ask the source about one pending game and move it to where it belongs.

    Returns the status and the path the payload now lives at, which is ``None``
    when the game is still pending in place.
    """
    payload = json.loads(input_path.read_text(encoding="utf-8"))

    raw_game_id = payload.get("gameId")
    if raw_game_id is None:
        raise ValueError(f"Missing gameId in {input_path}")
    game_id = f"{int(raw_game_id):010d}"

    attempts = int(payload.get(ATTEMPTS_KEY, 0))

    # A game that has already used up its attempts is quarantined without
    # spending another request on it.
    if attempts >= max_attempts and unresolved_dir is not None:
        return GameStatus.UNKNOWN, _quarantine(input_path, unresolved_dir, attempts)

    logger.info(f"Fetching final score for game {game_id} via '{source.name}'")
    result = source.fetch(game_id)

    if result is None:
        # The source is unreachable — that is not the game's fault, so this does
        # not count as an attempt.
        logger.info(f"Source gave no answer for {game_id}; leaving it pending")
        return GameStatus.UNKNOWN, None

    status = result.status

    if not status.is_terminal:
        if not _has_tipped_off(payload):
            # The source is right: the game has not started. Counting this would
            # make the attempt budget a function of how often the cycle is run,
            # so a few re-runs on the morning of a slate would quarantine the
            # whole evening. Leave the payload untouched.
            logger.info(
                f"Game {game_id} is {status.value} and has not tipped off yet; "
                "still pending (no attempt spent)"
            )
            return status, None

        attempts += 1
        payload[ATTEMPTS_KEY] = attempts
        _write_payload(payload, input_path)
        if attempts >= max_attempts and unresolved_dir is not None:
            return status, _quarantine(input_path, unresolved_dir, attempts)
        logger.info(
            f"Game {game_id} is {status.value}; still pending "
            f"(attempt {attempts}/{max_attempts})"
        )
        return status, None

    payload["status"] = status.value
    payload.pop(ATTEMPTS_KEY, None)

    if status is GameStatus.POSTPONED:
        if postponed_dir is None:
            logger.warning(
                f"Game {game_id} is postponed but no postponed directory was given; "
                "leaving it pending"
            )
            return status, None
        payload["postponed"] = 1
        destination = postponed_dir / input_path.name
        _write_payload(payload, destination)
        input_path.unlink()
        logger.info(f"Game {game_id} postponed; parked at {_display_path(destination)}")
        return status, destination

    payload["homeTeamFinalScore"] = result.home_score
    payload["awayTeamFinalScore"] = result.away_score
    payload["overtimes"] = result.overtimes
    payload["winner"] = resolve_winner_team_id(payload, result)
    payload["postponed"] = 0
    payload["attendance"] = result.attendance
    payload["inactivePlayers"] = result.inactive_players

    destination = output_dir / input_path.name
    _write_payload(payload, destination)

    # The game has been played; it is no longer "upcoming".
    input_path.unlink()

    logger.info(f"Saved enriched game to {_display_path(destination)}")
    return status, destination


def _quarantine(input_path: Path, unresolved_dir: Path, attempts: int) -> Path:
    """Park a game the source will not settle, so it stops blocking the cycle."""
    unresolved_dir.mkdir(parents=True, exist_ok=True)
    destination = unresolved_dir / input_path.name
    shutil.move(str(input_path), str(destination))
    logger.warning(
        "Quarantined %s after %d attempts without a terminal status. "
        "Investigate, then `make retry-unresolved` to put it back.",
        input_path.name,
        attempts,
    )
    return destination


def enrich_upcoming_games_results(
    input_dir: Path,
    output_dir: Path,
    source: ResultsSource | str = DEFAULT_SOURCE,
    delay_seconds: float = FETCH_DELAY_SECONDS,
    postponed_dir: Path | None = None,
    unresolved_dir: Path | None = None,
    max_attempts: int = MAX_FETCH_ATTEMPTS,
) -> dict[str, list[Path]]:
    """Resolve every pending game in ``input_dir``.

    Returns the paths written, grouped by what happened: ``final``,
    ``postponed``, ``quarantined``, and ``pending`` (still in ``input_dir``).
    """
    if isinstance(source, str):
        source = get_results_source(source)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if not source.available():
        raise RuntimeError(
            f"Results source '{source.name}' is not available. "
            "See its module docstring for setup."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    outcomes: dict[str, list[Path]] = {
        "final": [],
        "postponed": [],
        "quarantined": [],
        "pending": [],
    }

    pending = sorted(input_dir.glob("*.json"))
    for i, input_path in enumerate(pending):
        if i > 0 and delay_seconds:
            time.sleep(delay_seconds)
        status, written = enrich_upcoming_game_result(
            input_path,
            output_dir,
            source,
            postponed_dir=postponed_dir,
            unresolved_dir=unresolved_dir,
            max_attempts=max_attempts,
        )
        if written is None:
            outcomes["pending"].append(input_path)
        elif status is GameStatus.FINAL:
            outcomes["final"].append(written)
        elif status is GameStatus.POSTPONED:
            outcomes["postponed"].append(written)
        else:
            outcomes["quarantined"].append(written)

    logger.info(
        "Resolved %d pending game(s) using '%s': %d final, %d postponed, "
        "%d still pending, %d quarantined",
        len(pending),
        source.name,
        len(outcomes["final"]),
        len(outcomes["postponed"]),
        len(outcomes["pending"]),
        len(outcomes["quarantined"]),
    )
    return outcomes
