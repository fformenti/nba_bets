"""Merging the raw historical archive into the history table.

`make ingest-raw-games` used to write a separate seed CSV that only a first run
ever read, so `make full-rebuild` reparsed the raw archive and then silently threw
it away. Now the parse merges straight into the one history table, which makes the
precedence question load-bearing: the archive wins any gameId both hold, and every
game collected since the archive was last published has to survive.
"""

import pandas as pd
import pytest

from src.etl.ingestion.raw_games import merge_into_history

HOME, AWAY = 1610612741, 1610612748


def _row(game_id, date, home_score=100, away_score=90, **overrides):
    row = {
        "gameId": game_id,
        "gameDate": pd.Timestamp(date),
        "gameDateOnlyStr": date,
        "season": "2025/26",
        "hometeamId": HOME,
        "awayteamId": AWAY,
        "homeScore": home_score,
        "awayScore": away_score,
        "winner": HOME if home_score > away_score else AWAY,
        "postponed": 0,
        "attendance": 18000,
    }
    row.update(overrides)
    return row


def test_history_only_games_survive_the_merge():
    """The regression that matters: a rebuild must not drop collected games.

    The raw archive lags the live feed, so the games collected since it was last
    published exist nowhere else.
    """
    raw = pd.DataFrame([_row(22500001, "2025-10-22")])
    history = pd.DataFrame([_row(22500001, "2025-10-22"), _row(22500999, "2026-02-12")])

    combined = merge_into_history(raw, history)

    assert combined["gameId"].tolist() == [22500001, 22500999]


def test_the_raw_archive_wins_a_shared_game_id():
    """The archive is the curated upstream dump; a corrected score propagates."""
    raw = pd.DataFrame([_row(22500001, "2025-10-22", home_score=111, away_score=99)])
    history = pd.DataFrame([_row(22500001, "2025-10-22", home_score=0, away_score=0)])

    combined = merge_into_history(raw, history)

    assert len(combined) == 1
    assert combined.iloc[0]["homeScore"] == 111


def test_a_postponed_archive_row_never_beats_a_played_one():
    """22500529: the archive lags the league on rescheduled games.

    It still lists the original scoreless fixture long after the live feed has
    recorded the replay, and letting raw win would walk a played game back to
    postponed — dropping it from the regular-season table entirely.
    """
    raw = pd.DataFrame([_row(22500529, "2026-01-08", 0, 0, postponed=1, winner=0)])
    history = pd.DataFrame([_row(22500529, "2026-01-29", 113, 116)])

    combined = merge_into_history(raw, history)

    assert len(combined) == 1
    row = combined.iloc[0]
    assert row["postponed"] == 0
    assert row["homeScore"] == 113
    assert row["gameDateOnlyStr"] == "2026-01-29"


def test_a_postponed_archive_row_still_beats_a_postponed_history_row():
    """The exception is narrow: it only protects a genuinely played result."""
    raw = pd.DataFrame([_row(22500529, "2026-01-08", 0, 0, postponed=1, attendance=0)])
    history = pd.DataFrame([_row(22500529, "2026-01-08", 0, 0, postponed=1)])

    combined = merge_into_history(raw, history)

    assert len(combined) == 1
    assert combined.iloc[0]["attendance"] == 0


def test_an_absent_history_table_yields_the_parsed_archive():
    raw = pd.DataFrame([_row(22500001, "2025-10-22")])

    combined = merge_into_history(raw, pd.DataFrame())

    assert combined["gameId"].tolist() == [22500001]


def test_the_result_is_sorted_and_canonically_shaped():
    from src.etl.utils.common import CANONICAL_INGESTED_COLUMNS

    raw = pd.DataFrame([_row(22500003, "2026-01-05"), _row(22500001, "2025-10-22")])
    history = pd.DataFrame([_row(22500002, "2025-12-01")])

    combined = merge_into_history(raw, history)

    assert combined["gameId"].tolist() == [22500001, 22500002, 22500003]
    assert list(combined.columns) == CANONICAL_INGESTED_COLUMNS


def test_a_duplicate_game_id_refuses_to_write():
    """The guard is the last line of defence: every feature table counts games."""
    raw = pd.DataFrame([_row(22500001, "2025-10-22"), _row(22500001, "2025-10-22")])

    with pytest.raises(ValueError, match="duplicate gameId"):
        merge_into_history(raw, pd.DataFrame())
