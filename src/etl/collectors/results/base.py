"""The contract every source of finished-game outcomes must satisfy.

Retrieval is behind an interface because the provider is not settled yet. The
rest of the pipeline — enrichment, appending to history, accuracy scoring —
only knows about :class:`GameResult`, so swapping providers touches exactly one
file plus a line in the factory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class GameStatus(str, Enum):
    """What the source says has happened to a game.

    This is read from the provider's status field, never inferred from the
    scoreline. Inferring it is how a game that had simply not tipped off yet
    ended up recorded as permanently postponed: a 0-0 scoreline is the normal
    state of every game before it starts.

    ``UNKNOWN`` is the safe bucket. An ambiguous or unparseable status must
    leave the game pending rather than commit a guess to the history table.
    """

    FINAL = "final"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    POSTPONED = "postponed"
    UNKNOWN = "unknown"

    @property
    def is_terminal(self) -> bool:
        """Whether this status settles the game, one way or the other.

        Terminal statuses move the payload out of the pending directory. The
        rest leave it there to be asked about again.
        """
        return self in (GameStatus.FINAL, GameStatus.POSTPONED)


@dataclass
class GameResult:
    """State of a game as reported by a results source.

    ``status`` drives every routing decision downstream. ``postponed`` is kept
    because the historical backfill uses it as a column, and is derived from
    ``status`` rather than the other way round.
    """

    home_score: int = 0
    away_score: int = 0
    overtimes: int = 0
    winner: int = 0
    postponed: int = 0
    attendance: int | None = None
    inactive_players: list[dict[str, Any]] = field(default_factory=list)
    status: GameStatus = GameStatus.UNKNOWN


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
        """Outcome for ``game_id``, or ``None`` if the source cannot be reached.

        ``None`` means "ask again later" — a timeout, an outage. A game that
        simply has not been played yet is a successful answer with a non-final
        :class:`GameStatus`, not ``None``. Neither case may raise.
        """
        ...

    def available(self) -> bool:
        """Cheap check that this source is usable before a batch run."""
        ...
