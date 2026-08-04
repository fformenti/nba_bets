"""The contract every source of finished-game outcomes must satisfy.

Retrieval is behind an interface because the provider is not settled yet. The
rest of the pipeline — enrichment, appending to history, accuracy scoring —
only knows about :class:`GameResult`, so swapping providers touches exactly one
file plus a line in the factory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class GameResult:
    """Final state of a played game.

    ``postponed=1`` means the game did not happen; that is distinct from a
    source returning ``None``, which means "no outcome available yet".
    """

    home_score: int = 0
    away_score: int = 0
    overtimes: int = 0
    winner: int = 0
    postponed: int = 0
    attendance: int | None = None
    inactive_players: list[dict[str, Any]] = field(default_factory=list)


@runtime_checkable
class ResultsSource(Protocol):
    """Where finished-game outcomes come from.

    Implementations live alongside this module and are registered in
    ``__init__.py``. To add one — a bookmaker feed, a scraped scoreboard, a
    manual spreadsheet export — implement these two methods and register it.
    Nothing downstream changes.
    """

    name: str

    def fetch(self, game_id: str) -> GameResult | None:
        """Outcome for ``game_id``, or ``None`` if it is not available yet.

        ``None`` is the normal answer for a game that has not been played, and
        must not raise.
        """
        ...

    def available(self) -> bool:
        """Cheap check that this source is usable before a batch run."""
        ...
