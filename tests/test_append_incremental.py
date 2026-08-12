"""Folding fetched results into history, and what makes that safe to repeat.

The append used to rebuild the whole table from a frozen seed plus every result
file ever written. That made the inbox permanent storage — clearing it would
have erased every game collected since the seed — and it meant the step could
never be described as "add the new games". These tests pin the properties that
replaced it: the inbox is drained into an archive, re-running changes nothing,
and a game that comes back on a new date replaces its old row instead of
appearing twice.

Seeding the table from the raw historical archive belongs to `ingest_raw_games`
now — see tests/test_raw_games_merge.py — so the append only ever adds the inbox
to whatever history already exists.
"""

import json

import pandas as pd
import pytest

from src.etl.ingestion.append_games_results import add_game_results_to_historical
from src.etl.utils.common import game_type_from_id

HOME, AWAY = 1610612741, 1610612748


@pytest.fixture(autouse=True)
def _skip_location_enrichment(monkeypatch):
    """Location columns come from a lookup table that is not what is under test."""
    monkeypatch.setattr(
        "src.etl.ingestion.append_games_results.enrich_games_locations",
        lambda df: df,
    )


def _result_payload(game_id, date, home_score, away_score, **overrides):
    payload = {
        "gameId": game_id,
        "gameDate": f"{date}T20:00:00.000",
        "gameDateOnlyStr": date,
        "season": "2025/26",
        "homeTeamCity": "Atlanta",
        "homeTeamName": "Hawks",
        "hometeamId": HOME,
        "awayTeamCity": "Boston",
        "awayTeamName": "Celtics",
        "awayteamId": AWAY,
        "homeTeamFinalScore": home_score,
        "awayTeamFinalScore": away_score,
        "winner": HOME if home_score > away_score else AWAY,
        "overtimes": 0,
        "postponed": 0,
        "status": "final",
        "attendance": 18000,
        "inactivePlayers": [],
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def dirs(tmp_path):
    inbox = tmp_path / "results"
    archive = tmp_path / "archive"
    inbox.mkdir()
    return inbox, archive


@pytest.fixture
def history_path(tmp_path):
    """The existing history table the append reads, adds to, and rewrites."""
    path = tmp_path / "history.csv"
    pd.DataFrame(
        [
            {
                "gameId": 22500001,
                "gameDate": pd.Timestamp("2025-10-22"),
                "gameDateOnlyStr": "2025-10-22",
                "season": "2025/26",
                "hometeamId": HOME,
                "awayteamId": AWAY,
                "homeScore": 100,
                "awayScore": 90,
                "winner": HOME,
                "postponed": 0,
            }
        ]
    ).to_csv(path, index=False)
    return path


def _drop(inbox, payload):
    (inbox / f"{payload['gameId']}.json").write_text(json.dumps(payload))


def _append(dirs, output):
    inbox, archive = dirs
    return add_game_results_to_historical(
        upcoming_results_dir=inbox,
        output_path=output,
        archive_dir=archive,
    )


def test_new_results_are_appended_and_archived(dirs, history_path):
    inbox, archive = dirs
    _drop(inbox, _result_payload(22500002, "2026-02-20", 110, 105))

    combined = _append(dirs, history_path)

    assert sorted(combined["gameId"]) == [22500001, 22500002]
    assert (archive / "22500002.json").exists(), "consumed file must move to the archive"
    assert list(inbox.glob("*.json")) == [], "inbox must be drained"


def test_appending_twice_changes_nothing(dirs, history_path):
    """The inbox is drained, so a second run has nothing to add."""
    inbox = dirs[0]
    _drop(inbox, _result_payload(22500002, "2026-02-20", 110, 105))

    first = _append(dirs, history_path)
    second = _append(dirs, history_path)

    assert len(first) == len(second) == 2
    assert second["gameId"].duplicated().sum() == 0


def test_an_empty_inbox_leaves_history_alone(dirs, history_path):
    """A no-op run must rewrite nothing — the history table is the only copy."""
    before = history_path.read_text()

    result = _append(dirs, history_path)

    assert result["gameId"].tolist() == [22500001]
    assert history_path.read_text() == before


def test_a_missing_history_table_starts_from_the_inbox(dirs, tmp_path):
    """First run before any ingest: there is nothing to append onto yet."""
    inbox = dirs[0]
    output = tmp_path / "history.csv"
    _drop(inbox, _result_payload(22500002, "2026-02-20", 110, 105))

    combined = _append(dirs, output)

    assert combined["gameId"].tolist() == [22500002]


def test_a_rescheduled_game_replaces_its_old_row(dirs, history_path):
    """The bug that duplicated 22500529: the date moves, the gameId does not."""
    inbox = dirs[0]

    _drop(inbox, _result_payload(22500002, "2026-01-08", 0, 0, postponed=1, winner=0))
    _append(dirs, history_path)

    _drop(inbox, _result_payload(22500002, "2026-01-29", 113, 116))
    combined = _append(dirs, history_path)

    rows = combined[combined["gameId"] == 22500002]
    assert len(rows) == 1, "a replayed game must not appear twice"
    assert rows.iloc[0]["gameDateOnlyStr"] == "2026-01-29"
    assert rows.iloc[0]["homeScore"] == 113


def test_postponed_payloads_never_enter_history(dirs, history_path):
    inbox = dirs[0]
    _drop(
        inbox,
        _result_payload(22500002, "2026-02-20", 0, 0, status="postponed", postponed=1),
    )

    combined = _append(dirs, history_path)

    assert 22500002 not in combined["gameId"].tolist()
    assert (inbox / "22500002.json").exists(), "an unused file must not be archived"


@pytest.mark.parametrize(
    "game_id,expected",
    [
        (12500014, "Preseason"),
        (22500002, "Regular Season"),
        (42500101, "Playoffs"),
        (52500001, "Play-in Tournament"),
    ],
)
def test_game_type_is_derived_from_the_game_id(dirs, history_path, game_id, expected):
    """Results payloads carry no gameType, and the schedule does not mark rounds.

    Without this a playoff game collected in April would file as regular season.
    """
    inbox, _ = dirs
    _drop(inbox, _result_payload(game_id, "2026-04-20", 110, 105))

    combined = _append(dirs, history_path)

    row = combined[combined["gameId"] == game_id].iloc[0]
    assert row["gameType"] == expected == game_type_from_id(game_id)
