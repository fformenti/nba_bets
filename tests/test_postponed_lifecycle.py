"""A game's whole life through the incremental pipeline, postponements included.

Each piece had its own tests, but nothing ran a game from the schedule through
the selector, the results router, the postponed watch and the append in one
go, and the loop has never run for real since the project froze on
2026-01-26. These tests do, on tmp directories, with the real functions and a
hand-written results source.

The property under test is the one every feature depends on: a game lands in
history once, on the day it was actually played, in date order with the games
around it. A postponed game waits and comes back on its new date; a cancelled
game never comes back.
"""

import json

import pandas as pd
import pytest

from src.etl.collectors.postponed_watch import (
    reconcile_postponed_games,
    retry_unresolved_games,
)
from src.etl.collectors.results.placeholder_source import PlaceholderResultsSource
from src.etl.collectors.upcoming_games import _get_next_upcoming_games, _save_upcoming_games
from src.etl.collectors.upcoming_games_results import enrich_upcoming_games_results
from src.etl.ingestion.append_games_results import add_game_results_to_historical
from src.etl.ingestion.raw_games import merge_into_history

HOME, AWAY = 1610612741, 1610612748
A, B, C = 22500001, 22500002, 22500003


@pytest.fixture(autouse=True)
def _skip_location_enrichment(monkeypatch):
    """Location columns come from a lookup table that is not what is under test."""
    monkeypatch.setattr(
        "src.etl.ingestion.append_games_results.enrich_games_locations", lambda df: df
    )


class Pipeline:
    """The daily loop's state, in tmp directories, driven one step at a time."""

    def __init__(self, root):
        self.root = root
        self.dirs = {
            name: root / name
            for name in (
                "pending",
                "results",
                "archive",
                "postponed",
                "cancelled",
                "unresolved",
                "manual",
            )
        }
        for directory in self.dirs.values():
            directory.mkdir()
        self.history_path = root / "history.csv"
        self.schedule_path = root / "schedule.csv"
        self.tipoffs: dict[int, str] = {}

    # The league schedule, as fetch-league-schedule writes it.
    def schedule(self, **tipoffs):
        self.tipoffs.update({int(k[1:]): v for k, v in tipoffs.items()})
        pd.DataFrame(
            {
                "gameId": list(self.tipoffs),
                "gameDateTimeEst": [f"{t}:00Z" for t in self.tipoffs.values()],
            }
        ).to_csv(self.schedule_path, index=False)

    def seed_history(self, *rows):
        pd.DataFrame(
            [
                {
                    "gameId": game_id,
                    "gameDate": pd.Timestamp(tipoff),
                    "gameDateOnlyStr": tipoff[:10],
                    "season": "2025/26",
                    "hometeamId": HOME,
                    "awayteamId": AWAY,
                    "homeScore": 0 if postponed else 100,
                    "awayScore": 0 if postponed else 90,
                    "winner": 0 if postponed else HOME,
                    "postponed": postponed,
                    "gameType": "Regular Season",
                }
                for game_id, tipoff, postponed in rows
            ]
        ).to_csv(self.history_path, index=False)

    def history(self) -> pd.DataFrame:
        return pd.read_csv(self.history_path, parse_dates=["gameDate"])

    # fetch-upcoming-games: the selector, then the payloads it writes.
    def select(self) -> list[int]:
        schedule = pd.DataFrame(
            {
                "gameId": list(self.tipoffs),
                "tipoff": pd.to_datetime(list(self.tipoffs.values())),
                "hometeamId": HOME,
                "awayteamId": AWAY,
            }
        )
        schedule["gameDate"] = schedule["tipoff"].dt.date
        slate = _get_next_upcoming_games(
            schedule,
            self.history(),
            postponed_dir=self.dirs["postponed"],
            unresolved_dir=self.dirs["unresolved"],
            cancelled_dir=self.dirs["cancelled"],
        )
        payloads = pd.DataFrame(
            {
                "gameId": slate["gameId"],
                "gameDate": slate["tipoff"],
                "gameDateOnlyStr": slate["tipoff"].dt.strftime("%Y-%m-%d"),
                "season": "2025/26",
                "hometeamId": HOME,
                "homeTeamCity": "Chicago",
                "homeTeamName": "Bulls",
                "awayteamId": AWAY,
                "awayTeamCity": "Miami",
                "awayTeamName": "Heat",
            }
        )
        _save_upcoming_games(payloads, self.dirs["pending"])
        return sorted(int(path.stem) for path in self.dirs["pending"].glob("*.json"))

    # What the results source will say about a game.
    def source_says(self, game_id, **result):
        (self.dirs["manual"] / f"{game_id}.json").write_text(json.dumps(result))

    # fetch-game-results
    def fetch_results(self):
        return enrich_upcoming_games_results(
            self.dirs["pending"],
            self.dirs["results"],
            source=PlaceholderResultsSource(self.dirs["manual"]),
            delay_seconds=0,
            postponed_dir=self.dirs["postponed"],
            cancelled_dir=self.dirs["cancelled"],
            unresolved_dir=self.dirs["unresolved"],
        )

    # append-game-results
    def append(self):
        add_game_results_to_historical(
            self.dirs["results"], self.history_path, self.dirs["archive"]
        )

    # reconcile-postponed
    def reconcile(self):
        return reconcile_postponed_games(self.dirs["postponed"], self.schedule_path)

    def parked(self, name) -> list[int]:
        return sorted(int(path.stem) for path in self.dirs[name].glob("*.json"))


FINAL = {"homeTeamFinalScore": 113, "awayTeamFinalScore": 116}


@pytest.fixture
def loop(tmp_path):
    pipeline = Pipeline(tmp_path)
    pipeline.seed_history((A, "2026-01-05 19:30", 0))
    pipeline.schedule(
        g22500001="2026-01-05 19:30", g22500002="2026-01-08 20:00", g22500003="2026-01-10 19:00"
    )
    return pipeline


def test_a_postponed_game_returns_on_its_makeup_date_and_sorts_there(loop):
    # Jan 8: the game is called off.
    assert loop.select() == [B]
    loop.source_says(B, status="postponed")
    assert loop.fetch_results()["postponed"]
    assert loop.parked("postponed") == [B]

    # The loop moves on without it.
    assert loop.select() == [C]
    loop.source_says(C, **FINAL)
    loop.fetch_results()
    loop.append()

    # Nothing to release while the schedule still shows Jan 8.
    assert loop.reconcile() == []

    # The league reschedules it for Jan 29, behind the frontier's back.
    loop.schedule(g22500002="2026-01-29 20:00")
    assert loop.reconcile() == [B]
    assert loop.select() == [B]
    pending = json.loads((loop.dirs["pending"] / f"{B}.json").read_text())
    assert pending["gameDateOnlyStr"] == "2026-01-29"

    loop.source_says(B, **FINAL)
    loop.fetch_results()
    loop.append()

    history = loop.history()
    assert history["gameId"].tolist() == [A, C, B], "history is in played order"
    row = history[history["gameId"] == B].iloc[0]
    assert row["gameDateOnlyStr"] == "2026-01-29"
    assert row["postponed"] == 0
    assert loop.parked("postponed") == []


def test_a_postponed_archive_row_is_replaced_by_its_makeup_game(loop):
    """22500529: the archive froze it 0-0 on Jan 8; it was played on Jan 29."""
    loop.seed_history((A, "2026-01-05 19:30", 0), (B, "2026-01-08 20:00", 1))
    loop.schedule(g22500002="2026-01-29 20:00")

    assert loop.select() == [C]
    loop.source_says(C, **FINAL)
    loop.fetch_results()
    loop.append()

    assert loop.select() == [B]
    loop.source_says(B, **FINAL)
    loop.fetch_results()
    loop.append()

    history = loop.history()
    assert history["gameId"].tolist() == [A, C, B]
    assert history.set_index("gameId").loc[B, "gameDateOnlyStr"] == "2026-01-29"

    # A later `make ingest-raw-games` with the stale archive must not undo it.
    archive_row = history[history["gameId"] == B].assign(
        gameDate=pd.Timestamp("2026-01-08 20:00"),
        gameDateOnlyStr="2026-01-08",
        homeScore=0,
        awayScore=0,
        winner=0,
        postponed=1,
    )
    merged = merge_into_history(archive_row, history)
    assert merged.set_index("gameId").loc[B, "postponed"] == 0
    assert merged.set_index("gameId").loc[B, "gameDateOnlyStr"] == "2026-01-29"


def test_a_postponed_row_dated_ahead_does_not_move_the_frontier(loop):
    """The archive dates some 0-0 rows on their makeup day, past the real frontier.

    Counting such a row as the frontier put Jan 10 more than the grace period
    behind it, so the game would have been quarantined instead of collected.
    """
    loop.seed_history((A, "2026-01-05 19:30", 0), (B, "2026-02-15 20:00", 1))
    loop.schedule(g22500002="2026-02-15 20:00")

    assert loop.select() == [C]
    assert loop.parked("unresolved") == []


def test_the_source_s_played_date_beats_the_schedule_s(loop):
    """The schedule was stale; the source knows the game went a day late."""
    assert loop.select() == [B]
    loop.source_says(B, gameDate="2026-01-09", **FINAL)
    loop.fetch_results()
    loop.append()

    row = loop.history().set_index("gameId").loc[B]
    assert row["gameDateOnlyStr"] == "2026-01-09"
    assert row["gameDate"] == pd.Timestamp("2026-01-09 20:00"), "the tip-off time is kept"
    archived = json.loads((loop.dirs["archive"] / f"{B}.json").read_text())
    assert archived["scheduledGameDate"] == "2026-01-08"


def test_a_source_that_has_the_game_on_a_later_date_parks_it(loop):
    """No status says postponed, but a game still to come after its tip-off was."""
    assert loop.select() == [B]
    loop.source_says(B, status="scheduled", gameDate="2026-01-29")

    assert loop.fetch_results()["postponed"]
    assert loop.parked("postponed") == [B]


def test_a_cancelled_game_is_recorded_and_never_comes_back(loop):
    assert loop.select() == [B]
    loop.source_says(B, status="cancelled")

    assert loop.fetch_results()["cancelled"]
    assert loop.parked("cancelled") == [B]
    assert loop.parked("postponed") == []

    assert loop.select() == [C], "a cancelled game is not owed a fixture"
    loop.source_says(C, **FINAL)
    loop.fetch_results()
    loop.append()
    assert B not in loop.history()["gameId"].tolist()
    assert loop.reconcile() == [], "nothing watches a cancelled game"


def test_a_new_tip_off_on_the_same_day_releases_a_parked_game(loop):
    """A suspended game resumed that evening, or a game pushed back a few hours."""
    loop.select()
    loop.source_says(B, status="postponed")
    loop.fetch_results()

    loop.schedule(g22500002="2026-01-08 23:00")

    assert loop.reconcile() == [B]


def test_a_game_off_the_schedule_stays_parked(loop):
    loop.select()
    loop.source_says(B, status="postponed")
    loop.fetch_results()

    del loop.tipoffs[B]
    loop.schedule()

    assert loop.reconcile() == []
    assert loop.parked("postponed") == [B]


def test_retry_releases_quarantined_games(loop):
    (loop.dirs["unresolved"] / f"{B}.json").write_text("{}")

    assert retry_unresolved_games(loop.dirs["unresolved"]) == [B]
    assert loop.select() == [B]
