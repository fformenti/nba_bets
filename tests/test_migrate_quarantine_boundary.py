"""Which pending games the layout migration is allowed to park.

The migration quarantines pending games the pipeline has already moved past,
which is right for a game history has left behind and wrong for the slate
currently in flight. The selector deliberately finishes a partially-collected
day before moving on, so the games being worked on normally sit *on* the
frontier date — parking those made a second `make migrate-incremental` throw
away a healthy queue.
"""

import json

import pandas as pd
import pytest

from src.etl.ingestion.migrate_incremental_layout import quarantine_abandoned_pending

FRONTIER = pd.Timestamp("2026-02-12")


@pytest.fixture
def dirs(tmp_path):
    pending = tmp_path / "upcoming_games"
    unresolved = tmp_path / "unresolved"
    pending.mkdir()
    return pending, unresolved


def _pending(pending_dir, game_id: int, date: str):
    path = pending_dir / f"{game_id}.json"
    path.write_text(json.dumps({"gameId": game_id, "gameDate": f"{date}T19:00:00.000"}))
    return path


def test_a_game_behind_the_frontier_is_parked(dirs):
    pending, unresolved = dirs
    path = _pending(pending, 22500001, "2026-02-11")

    moved = quarantine_abandoned_pending(pending, unresolved, FRONTIER)

    assert [p.name for p in moved] == [path.name]
    assert not path.exists()
    assert (unresolved / path.name).exists()


def test_the_slate_on_the_frontier_is_left_alone(dirs):
    """The in-flight slate sits on the frontier; it has not been passed by."""
    pending, unresolved = dirs
    path = _pending(pending, 22500002, "2026-02-12")

    assert quarantine_abandoned_pending(pending, unresolved, FRONTIER) == []
    assert path.exists()


def test_running_twice_parks_nothing_extra(dirs):
    """Re-running the migration must not eat the queue it just left in place."""
    pending, unresolved = dirs
    _pending(pending, 22500002, "2026-02-12")
    _pending(pending, 22500003, "2026-02-13")
    _pending(pending, 22500001, "2026-02-11")

    first = quarantine_abandoned_pending(pending, unresolved, FRONTIER)
    second = quarantine_abandoned_pending(pending, unresolved, FRONTIER)

    assert len(first) == 1
    assert second == []
    assert sorted(p.name for p in pending.glob("*.json")) == [
        "22500002.json",
        "22500003.json",
    ]


def test_a_payload_with_no_usable_date_is_left_alone(dirs):
    pending, unresolved = dirs
    path = pending / "22500004.json"
    path.write_text(json.dumps({"gameId": 22500004, "gameDate": None}))

    assert quarantine_abandoned_pending(pending, unresolved, FRONTIER) == []
    assert path.exists()
