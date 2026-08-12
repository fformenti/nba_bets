"""TODO: placeholder results source — replace once a provider is chosen.

This exists so the full historical + incremental loop (predict → play → score →
fold into history) can be exercised end to end today, without depending on a
live feed. It reads outcomes that you drop by hand into
``data/raw/incremental/manual_results/`` as ``{gameId}.json``:

.. code-block:: json

    {
      "homeTeamFinalScore": 112,
      "awayTeamFinalScore": 99,
      "overtimes": 0
    }

``gameId`` may be written with or without leading zeros. Games with no file
return ``None``, which the collector treats as "not played yet" — exactly how a
real source behaves mid-slate.

To exercise the other outcomes, set ``"status"`` explicitly to any
:class:`~src.etl.collectors.results.base.GameStatus` value:

.. code-block:: json

    {"status": "postponed"}
    {"status": "in_progress", "homeTeamFinalScore": 54, "awayTeamFinalScore": 61}

**Replacing this:** implement :class:`~src.etl.collectors.results.base.ResultsSource`
in a new module here and register it in ``__init__.py``. No caller changes —
``src/cli/fetch_game_results.py`` picks the source by name.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config.paths import MANUAL_RESULTS_DIR
from src.etl.collectors.results.base import GameResult, GameStatus
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class PlaceholderResultsSource:
    """Read hand-dropped outcome files from a local directory."""

    name = "placeholder"

    def __init__(self, results_dir: Path | None = None):
        self.results_dir = Path(results_dir or MANUAL_RESULTS_DIR)

    def _candidate_paths(self, game_id: str) -> list[Path]:
        """Accept the id zero-padded or not — both spellings occur upstream."""
        stripped = game_id.lstrip("0") or "0"
        names = {game_id, stripped, f"{int(stripped):010d}"}
        return [self.results_dir / f"{name}.json" for name in names]

    def fetch(self, game_id: str) -> GameResult | None:
        path = next((p for p in self._candidate_paths(game_id) if p.exists()), None)
        if path is None:
            logger.info(f"No manual result for game {game_id} yet")
            return None

        payload = json.loads(path.read_text(encoding="utf-8"))
        status = self._read_status(path, payload)

        if status is not GameStatus.FINAL:
            logger.info(f"Manual result for {game_id}: {status.value}")
            return GameResult(
                postponed=1 if status is GameStatus.POSTPONED else 0,
                status=status,
            )

        try:
            home_score = int(payload["homeTeamFinalScore"])
            away_score = int(payload["awayTeamFinalScore"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{path} must contain integer 'homeTeamFinalScore' and "
                f"'awayTeamFinalScore' fields. Got: {payload}"
            ) from exc

        logger.info(f"Manual result for {game_id}: {home_score}-{away_score}")
        return GameResult(
            home_score=home_score,
            away_score=away_score,
            overtimes=int(payload.get("overtimes", 0)),
            # Left at 0: `winner` is the winning *teamId*, and the team ids live
            # on the game payload, not in the manual file. The enrichment step
            # fills it in — see resolve_winner_team_id().
            winner=0,
            postponed=0,
            attendance=payload.get("attendance"),
            inactive_players=payload.get("inactivePlayers", []),
            status=GameStatus.FINAL,
        )

    @staticmethod
    def _read_status(path: Path, payload: dict) -> GameStatus:
        """An explicit status wins; otherwise a dropped file means a final score.

        ``postponed: true`` is still honoured so files written before statuses
        existed keep working.
        """
        raw_status = payload.get("status")
        if raw_status is not None:
            try:
                return GameStatus(str(raw_status).lower())
            except ValueError as exc:
                raise ValueError(
                    f"{path} has unknown status {raw_status!r}. Expected one of: "
                    f"{', '.join(s.value for s in GameStatus)}"
                ) from exc
        if payload.get("postponed"):
            return GameStatus.POSTPONED
        return GameStatus.FINAL

    def available(self) -> bool:
        if not self.results_dir.exists():
            logger.warning(
                f"{self.results_dir} does not exist. Create it and drop "
                "'{gameId}.json' files there to use the placeholder source."
            )
            return False
        return True
