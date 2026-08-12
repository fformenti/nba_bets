"""Which games get collected next, and why the queue cannot jam.

The pipeline used to deadlock on a single game. It asked "what is still missing
from the last played date?", and if the results source never answered for one
fixture, that question had the same answer forever: the frontier could not move
past a date that still looked incomplete, so every later game was unreachable.

These tests pin the two properties that fix it — a game that cannot be settled
is eventually parked, and selection is driven by what history is missing rather
than by how far the frontier has travelled.
"""

import json

import pandas as pd
import pytest

from src.config.constants import FETCH_ATTEMPTS_KEY
from src.etl.collectors.upcoming_games import (
    _get_next_upcoming_games,
    _save_upcoming_games,
)

HOME, AWAY = 1610612741, 1610612748


def _schedule(*rows) -> pd.DataFrame:
    """A schedule frame shaped like the one _load_schedule produces."""
    return pd.DataFrame(
        [
            {
                "gameId": game_id,
                "gameDate": pd.Timestamp(date).date(),
                "hometeamId": HOME,
                "awayteamId": AWAY,
            }
            for game_id, date in rows
        ]
    )


def _history(*rows) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gameId": game_id,
                "gameDate": pd.Timestamp(date),
                "gameDateOnlyStr": date,
                "hometeamId": HOME,
                "awayteamId": AWAY,
                "postponed": postponed,
            }
            for game_id, date, postponed in rows
        ]
    )


@pytest.fixture
def parked_dirs(tmp_path):
    postponed = tmp_path / "postponed"
    unresolved = tmp_path / "unresolved"
    postponed.mkdir()
    unresolved.mkdir()
    return postponed, unresolved


def test_partially_collected_day_is_finished_first(parked_dirs):
    """The earliest incomplete date is the next slate, not the following day."""
    postponed_dir, unresolved_dir = parked_dirs
    schedule = _schedule((1, "2026-02-19"), (2, "2026-02-19"), (3, "2026-02-20"))
    history = _history((1, "2026-02-19", 0))

    upcoming = _get_next_upcoming_games(
        schedule, history, postponed_dir=postponed_dir, unresolved_dir=unresolved_dir
    )

    assert upcoming["gameId"].tolist() == [2]


def test_a_parked_game_does_not_pin_the_slate(parked_dirs):
    """The deadlock, directly: a quarantined game must not hold the frontier."""
    postponed_dir, unresolved_dir = parked_dirs
    (unresolved_dir / "2.json").write_text(json.dumps({"gameId": 2}))

    schedule = _schedule((1, "2026-02-19"), (2, "2026-02-19"), (3, "2026-02-20"))
    history = _history((1, "2026-02-19", 0))

    upcoming = _get_next_upcoming_games(
        schedule, history, postponed_dir=postponed_dir, unresolved_dir=unresolved_dir
    )

    assert upcoming["gameId"].tolist() == [3], "should have advanced past the parked game"


def test_postponed_game_being_watched_is_not_re_emitted(parked_dirs):
    postponed_dir, unresolved_dir = parked_dirs
    (postponed_dir / "2.json").write_text(json.dumps({"gameId": 2}))

    schedule = _schedule((1, "2026-02-19"), (2, "2026-02-19"), (3, "2026-02-20"))
    history = _history((1, "2026-02-19", 0))

    upcoming = _get_next_upcoming_games(
        schedule, history, postponed_dir=postponed_dir, unresolved_dir=unresolved_dir
    )

    assert upcoming["gameId"].tolist() == [3]


def test_makeup_game_behind_the_frontier_is_reachable(parked_dirs):
    """A game rescheduled into the past must still be collected.

    Asking only for dates after the frontier could never see this one.
    """
    postponed_dir, unresolved_dir = parked_dirs
    schedule = _schedule((1, "2026-02-19"), (2, "2026-02-18"))
    history = _history((1, "2026-02-19", 0))

    upcoming = _get_next_upcoming_games(
        schedule,
        history,
        postponed_dir=postponed_dir,
        unresolved_dir=unresolved_dir,
        grace_days=5,
    )

    assert upcoming["gameId"].tolist() == [2]


def test_a_postponed_history_row_still_owes_its_fixture(parked_dirs):
    """A postponed row records that nothing happened, so the game is still due."""
    postponed_dir, unresolved_dir = parked_dirs
    schedule = _schedule((1, "2026-03-18"))
    history = _history((1, "2026-01-25", 1), (9, "2026-02-19", 0))

    upcoming = _get_next_upcoming_games(
        schedule, history, postponed_dir=postponed_dir, unresolved_dir=unresolved_dir
    )

    assert upcoming["gameId"].tolist() == [1]


def test_stragglers_past_the_grace_window_are_quarantined(parked_dirs):
    postponed_dir, unresolved_dir = parked_dirs
    schedule = _schedule((1, "2026-02-01"), (2, "2026-02-20"))
    history = _history((9, "2026-02-19", 0))

    upcoming = _get_next_upcoming_games(
        schedule,
        history,
        postponed_dir=postponed_dir,
        unresolved_dir=unresolved_dir,
        grace_days=3,
    )

    assert upcoming["gameId"].tolist() == [2]
    assert (unresolved_dir / "1.json").exists()


def test_nothing_left_to_collect_returns_empty(parked_dirs):
    postponed_dir, unresolved_dir = parked_dirs
    schedule = _schedule((1, "2026-02-19"))
    history = _history((1, "2026-02-19", 0))

    upcoming = _get_next_upcoming_games(
        schedule, history, postponed_dir=postponed_dir, unresolved_dir=unresolved_dir
    )

    assert upcoming.empty


class TestPendingPayloadRefresh:
    """What happens to a payload already sitting in the pending directory.

    A payload used to be written once and never touched again, so a game the
    league rescheduled kept its original tip-off — and that stale date is what
    ``append-game-results`` wrote into history.
    """

    @staticmethod
    def _pending_dir(tmp_path):
        d = tmp_path / "upcoming_games"
        d.mkdir()
        return d

    def test_a_rescheduled_game_gets_the_new_date(self, tmp_path):
        pending = self._pending_dir(tmp_path)
        _save_upcoming_games(_schedule((1, "2026-02-19")), pending)

        _save_upcoming_games(_schedule((1, "2026-02-21")), pending)

        assert json.loads((pending / "1.json").read_text())["gameDate"].startswith(
            "2026-02-21"
        )

    def test_the_attempt_counter_survives_a_refresh(self, tmp_path):
        """The schedule owns the fixture; the collector owns the bookkeeping."""
        pending = self._pending_dir(tmp_path)
        _save_upcoming_games(_schedule((1, "2026-02-19")), pending)
        payload = json.loads((pending / "1.json").read_text())
        payload[FETCH_ATTEMPTS_KEY] = 3
        (pending / "1.json").write_text(json.dumps(payload))

        _save_upcoming_games(_schedule((1, "2026-02-21")), pending)

        refreshed = json.loads((pending / "1.json").read_text())
        assert refreshed[FETCH_ATTEMPTS_KEY] == 3
        assert refreshed["gameDate"].startswith("2026-02-21")

    def test_an_unchanged_slate_is_not_rewritten(self, tmp_path):
        """Re-running the collector must not churn the queue's mtimes."""
        pending = self._pending_dir(tmp_path)
        schedule = _schedule((1, "2026-02-19"), (2, "2026-02-19"))
        _save_upcoming_games(schedule, pending)
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in pending.glob("*")}

        _save_upcoming_games(schedule, pending)

        after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in pending.glob("*")}
        assert after == before

    def test_a_game_no_longer_in_the_slate_is_pruned(self, tmp_path):
        pending = self._pending_dir(tmp_path)
        _save_upcoming_games(_schedule((1, "2026-02-19"), (2, "2026-02-19")), pending)

        _save_upcoming_games(_schedule((1, "2026-02-19")), pending)

        assert sorted(p.name for p in pending.glob("*.json")) == ["1.json"]
