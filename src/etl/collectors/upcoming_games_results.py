"""Stamp fetched outcomes onto the upcoming-game JSON files.

Where the outcome comes from is decided by the caller: this module takes any
:class:`~src.etl.collectors.results.base.ResultsSource` and knows nothing about
nba_api, scraping, or hand-entered files.

Enriched payloads land in ``data/raw/incremental/upcoming_games_results/`` and
the source file is removed from ``upcoming_games/``, so the two directories
together answer "what is still pending?" and "what has been played?".
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.config.paths import PROJECT_ROOT
from src.etl.collectors.results import DEFAULT_SOURCE, get_results_source
from src.etl.collectors.results.base import ResultsSource
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# stats.nba.com rate-limits aggressively; pace batch requests.
FETCH_DELAY_SECONDS = 4


def _display_path(path: Path) -> str:
    """Project-relative path for logging, falling back to the absolute one.

    ``--output-dir`` may point anywhere, and a log line must never be the thing
    that fails a run.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


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
) -> Path | None:
    """Attach the final score to one game payload and move it to ``output_dir``.

    Returns the written path, or ``None`` when the source has no outcome yet —
    in which case the input file is left in place for the next run.
    """
    payload = json.loads(input_path.read_text(encoding="utf-8"))

    raw_game_id = payload.get("gameId")
    if raw_game_id is None:
        raise ValueError(f"Missing gameId in {input_path}")
    game_id = f"{int(raw_game_id):010d}"

    logger.info(f"Fetching final score for game {game_id} via '{source.name}'")
    result = source.fetch(game_id)

    if result is None:
        logger.info(f"No result for {game_id} yet; leaving it pending")
        return None

    payload["homeTeamFinalScore"] = result.home_score
    payload["awayTeamFinalScore"] = result.away_score
    payload["overtimes"] = result.overtimes
    payload["winner"] = resolve_winner_team_id(payload, result)
    payload["postponed"] = result.postponed
    payload["attendance"] = result.attendance
    payload["inactivePlayers"] = result.inactive_players

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / input_path.name
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)

    # The game has been played; it is no longer "upcoming".
    input_path.unlink()

    logger.info(f"Saved enriched game to {_display_path(output_path)}")
    return output_path


def enrich_upcoming_games_results(
    input_dir: Path,
    output_dir: Path,
    source: ResultsSource | str = DEFAULT_SOURCE,
    delay_seconds: float = FETCH_DELAY_SECONDS,
) -> list[Path]:
    """Enrich every pending game in ``input_dir``. Returns the paths written."""
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

    written: list[Path] = []
    pending = sorted(input_dir.glob("*.json"))
    for i, input_path in enumerate(pending):
        if i > 0 and delay_seconds:
            time.sleep(delay_seconds)
        output_path = enrich_upcoming_game_result(input_path, output_dir, source)
        if output_path is not None:
            written.append(output_path)

    logger.info(
        f"Enriched {len(written)} of {len(pending)} pending game(s) using '{source.name}'"
    )
    return written
