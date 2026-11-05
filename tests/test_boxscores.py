"""Box score fetcher: refresh rule, schema guard, what is stored, validation counts.

Nothing touches the network. Downloads go through a stub session serving
in-memory parquet, and every path points under tmp_path.
"""

import dataclasses
import io
from datetime import date

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import requests

from src.etl.collectors.boxscores import (
    SCHEDULE,
    TEAM,
    fetch_boxscores,
    needs_download,
    prepare_season,
    unplayed_games,
)
from src.etl.collectors.boxscores_validation import validate_season
from src.etl.collectors.sportsdataverse import (
    SchemaDriftError,
    check_schema,
    current_season_end_year,
    season_end_year,
    season_label,
)
from src.etl.utils.common import atomic_write_parquet

_SAMPLE_VALUES = {"int64": 1, "int32": 0, "double": 0.5, "string": "x"}


def _team_table(end_year: int, schema: dict[str, str] = TEAM.schema) -> pa.Table:
    """A two-row team file for one game, typed exactly as the release is."""
    columns = {}
    for name, type_name in schema.items():
        value = _SAMPLE_VALUES[type_name]
        if name == "season":
            value = end_year
        elif name == "game_id":
            value = f"002{str(end_year - 1)[-2:]}00001"
        arrow_type = pa.string() if type_name == "string" else pa.type_for_alias(type_name)
        columns[name] = pa.array([value, value], type=arrow_type)
    columns["side"] = pa.array(["home", "away"])
    return pa.table(columns)


def _game_id(end_year: int, number: int = 1) -> str:
    return f"002{str(end_year - 1)[-2:]}{number:05d}"


def _schedule_frame(end_year: int, games) -> pd.DataFrame:
    """Schedule rows as the release has them: ``games`` is (number, date, played)."""
    rows = []
    for number, game_date, played in games:
        rows.append(
            {
                "game_id": _game_id(end_year, number),
                "season": end_year,
                "season_type": "regular-season",
                "game_date": game_date,
                "matchup": "AAA vs. BBB",
                "home_team_id": 1,
                "home_team_abbreviation": "AAA",
                "home_team_name": "Home",
                "home_pts": 100 if played else None,
                "home_wl": "W" if played else None,
                "away_team_id": 2,
                "away_team_abbreviation": "BBB",
                "away_team_name": "Away",
                "away_pts": 90 if played else None,
                "away_wl": "L" if played else None,
            }
        )
    return pd.DataFrame(rows)


def _schedule_table(end_year: int, games) -> pa.Table:
    frame = _schedule_frame(end_year, games)
    schema = pa.schema(
        [
            (name, pa.string() if type_name == "string" else pa.type_for_alias(type_name))
            for name, type_name in SCHEDULE.schema.items()
        ]
    )
    return pa.Table.from_pandas(frame, schema=schema, preserve_index=False)


def _parquet_bytes(table: pa.Table) -> bytes:
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


# ── Season keys ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("end_year", "label"), [(1997, "1996/97"), (2000, "1999/00"), (2027, "2026/27")]
)
def test_the_release_end_year_maps_to_our_season_label(end_year, label):
    assert season_label(end_year) == label


@pytest.mark.parametrize(
    ("today", "end_year"), [(date(2026, 10, 25), 2027), (date(2027, 3, 1), 2027)]
)
def test_the_current_season_spans_the_new_year(today, end_year):
    assert current_season_end_year(today) == end_year


@pytest.mark.parametrize(
    ("game_id", "end_year"), [("0029600001", 1997), ("0021201214", 2013), (22500529, 2026)]
)
def test_the_season_is_read_from_the_game_id(game_id, end_year):
    assert season_end_year(game_id) == end_year


# ── Refresh rule ─────────────────────────────────────────────────────────────


def _decide(**overrides):
    kwargs = {
        "stored": True,
        "end_year": 2027,
        "current_end_year": 2027,
        "stored_timestamp": "t1",
        "release_timestamp": "t1",
        "refresh_all": False,
    }
    return needs_download(**{**kwargs, **overrides})


def test_a_missing_season_is_always_downloaded():
    assert _decide(stored=False, end_year=1997)


def test_the_current_season_is_refetched_when_the_release_moved():
    assert _decide(release_timestamp="t2")


def test_the_current_season_is_skipped_when_the_release_has_not_moved():
    assert not _decide()


def test_a_past_season_is_skipped_although_the_release_moved():
    assert not _decide(end_year=2020, release_timestamp="t2")


def test_refresh_all_refetches_a_past_season_when_the_release_moved():
    assert _decide(end_year=2020, release_timestamp="t2", refresh_all=True)


# ── Schema guard ─────────────────────────────────────────────────────────────


def test_the_published_schema_passes():
    check_schema(_team_table(2025), TEAM, 2025)


def test_a_missing_column_is_drift():
    with pytest.raises(SchemaDriftError, match="missing=\\['blocks'\\]"):
        check_schema(_team_table(2025).drop_columns(["blocks"]), TEAM, 2025)


def test_an_added_column_is_drift():
    table = _team_table(2025).append_column("pace", pa.array([1.0, 1.0]))
    with pytest.raises(SchemaDriftError, match="added=\\['pace'\\]"):
        check_schema(table, TEAM, 2025)


def test_a_retyped_column_is_drift():
    schema = {**TEAM.schema, "points": "double"}
    with pytest.raises(SchemaDriftError, match="points"):
        check_schema(_team_table(2025, schema), TEAM, 2025)


def test_a_file_holding_another_season_is_drift():
    with pytest.raises(SchemaDriftError, match="expected 2026"):
        check_schema(_team_table(2025), TEAM, 2026)


# ── Fetch, end to end against a stub ────────────────────────────────────────


class _StubSession:
    def __init__(self, files: dict[str, bytes]):
        self.files = files
        self.requested: list[str] = []

    def get(self, url, timeout):
        assert timeout, "every request needs a timeout"
        self.requested.append(url)
        response = requests.Response()
        response.url = url
        name = url.rsplit("/", 1)[-1]
        response.status_code = 200 if name in self.files else 404
        response._content = self.files.get(name, b"")
        return response


TODAY = date(2026, 11, 4)


def _stub(timestamp: str, seasons: list[int], schedule_games=None) -> _StubSession:
    """Release files for ``seasons``; game 1 of each was played on its season's opener."""
    files = {"timestamp.txt": timestamp.encode()}
    for year in seasons:
        files[TEAM.filename(year)] = _parquet_bytes(_team_table(year))
        games = (schedule_games or {}).get(year, [(1, f"{year - 1}-10-25", True)])
        files[SCHEDULE.filename(year)] = _parquet_bytes(_schedule_table(year, games))
    return _StubSession(files)


def _fetch(tmp_path, session, seasons, **kwargs):
    kind = dataclasses.replace(TEAM, directory=tmp_path / "team")
    return fetch_boxscores(
        seasons,
        kinds=(kind,),
        manifest_path=tmp_path / "manifest.json",
        unplayed_path=tmp_path / "unplayed.csv",
        session=session,
        today=TODAY,
        **kwargs,
    )


def _actions(outcomes):
    return {outcome.end_year: outcome.action for outcome in outcomes}


def test_a_backfill_stores_every_season_with_our_label(tmp_path):
    outcomes = _fetch(tmp_path, _stub("t1", [2026, 2027]), [2026, 2027])

    assert _actions(outcomes) == {2026: "downloaded", 2027: "downloaded"}
    stored = pd.read_parquet(tmp_path / "team" / TEAM.filename(2026))
    assert stored["season_label"].unique().tolist() == ["2025/26"]
    assert stored["season"].unique().tolist() == [2026]
    assert stored["game_date"].dt.strftime("%Y-%m-%d").unique().tolist() == ["2025-10-25"]


def test_a_second_run_downloads_only_the_current_season_once_the_release_moves(tmp_path):
    _fetch(tmp_path, _stub("t1", [2026, 2027]), [2026, 2027])

    unchanged = _stub("t1", [2026, 2027])
    assert _actions(_fetch(tmp_path, unchanged, [2026, 2027])) == {
        2026: "skipped",
        2027: "skipped",
    }
    assert unchanged.requested == [TEAM.timestamp_url]

    moved = _stub("t2", [2026, 2027])
    assert _actions(_fetch(tmp_path, moved, [2026, 2027])) == {
        2026: "skipped",
        2027: "downloaded",
    }


def test_an_unpublished_current_season_is_skipped_but_a_missing_past_one_fails(tmp_path):
    assert _actions(_fetch(tmp_path, _stub("t1", [2026]), [2026, 2027]))[2027] == "not_published"

    with pytest.raises(requests.HTTPError):
        _fetch(tmp_path, _stub("t1", [2027]), [2025])


def test_drift_stops_the_write(tmp_path):
    session = _stub("t1", [2027])
    session.files[TEAM.filename(2027)] = _parquet_bytes(_team_table(2027).drop_columns(["blocks"]))

    with pytest.raises(SchemaDriftError):
        _fetch(tmp_path, session, [2027])
    assert not (tmp_path / "team" / TEAM.filename(2027)).exists()
    assert not (tmp_path / "manifest.json").exists()


def test_the_atomic_parquet_write_keeps_leading_zeros(tmp_path):
    path = tmp_path / "out.parquet"
    atomic_write_parquet(pd.DataFrame({"game_id": ["0022400001"]}), path)

    assert pd.read_parquet(path)["game_id"].tolist() == ["0022400001"]
    assert [p.name for p in tmp_path.iterdir()] == ["out.parquet"]


# ── Validation counts ────────────────────────────────────────────────────────


def _team_rows(game_id, home_points, away_points, sides=("home", "away"), day="2025-01-10"):
    return [
        {
            "game_id": game_id,
            "team_id": team_id,
            "side": side,
            "points": points,
            "season_type_id": "2",
            "game_date": pd.Timestamp(day),
        }
        for team_id, side, points in [(1, sides[0], home_points), (2, sides[1], away_points)]
    ]


def _player_rows(game_id, home_points, away_points):
    return [
        {"game_id": game_id, "team_id": 1, "points": home_points - 10},
        {"game_id": game_id, "team_id": 1, "points": 10},
        {"game_id": game_id, "team_id": 2, "points": away_points},
    ]


def test_validation_counts_each_kind_of_defect():
    team = pd.DataFrame(
        _team_rows("0022400001", 100, 90)  # clean
        + _team_rows("0022400002", 100, 90, sides=("home", "home"))
        + _team_rows("0022400003", 100, 90)  # players disagree
        + _team_rows("0022400004", 100, 90)  # no player rows yet
        + _team_rows("0022400005", 100, 90, day="2025-02-19")  # a makeup game, no player rows
        + [_team_rows("0022400006", 100, 90)[0]]  # one team row only
    )
    player = pd.DataFrame(
        _player_rows("0022400001", 100, 90)
        + _player_rows("0022400002", 100, 90)
        + _player_rows("0022400003", 100, 89)
        + _player_rows("0022400006", 100, 90)[:2]
    )
    history = pd.DataFrame(
        {
            "game_id": ["0022400001", "0022400002", "0022400005", "0022400099", "0042300101"],
            # 0022400005 still carries the date it was postponed from.
            "game_date": pd.to_datetime(
                ["2025-01-10", "2025-01-10", "2025-01-09", "2025-01-10", "2024-05-01"]
            ),
        }
    )

    counts = validate_season(
        2025, team, player, history, all_box_game_ids=frozenset({"0042300101"})
    )

    assert counts["games"] == 6
    assert counts["games_not_two_team_rows"] == 1
    assert counts["games_without_home_and_away"] == 2  # the two-home game and the one-row game
    assert counts["points_mismatch_games"] == 1
    assert counts["games_team_only"] == 2  # games 4 and 5
    assert counts["history_found"] == 3
    assert counts["history_coverage"] == 0.6
    assert counts["history_in_other_box_season"] == 1
    assert counts["box_games_not_in_history"] == 3
    assert counts["history_date_mismatches"] == 1
    assert counts["last_game_date"] == date(2025, 2, 19)


# ── What is stored: played games only, on the day they were played ─────────


def _box_rows(end_year, number, points=100):
    return [
        {"game_id": _game_id(end_year, number), "side": side, "points": points}
        for side in ("home", "away")
    ]


def test_only_played_games_with_a_past_date_are_stored():
    frame = pd.DataFrame(
        _box_rows(2025, 1)  # played on its scheduled date
        + _box_rows(2025, 2)  # postponed from January, played in February
        + _box_rows(2025, 3, points=None)  # placeholder with null stats
        + _box_rows(2025, 4)  # the schedule lists no result for it
        + _box_rows(2025, 5)  # dated after today
        + _box_rows(2025, 6)  # not on the schedule at all
    )
    schedule = _schedule_frame(
        2025,
        [
            (1, "2025-01-09", True),
            (2, "2025-02-19", True),
            (3, "2025-03-01", False),
            (4, "2025-03-02", False),
            (5, "2025-04-30", True),
        ],
    )

    kept, counts = prepare_season(frame, schedule, 2025, today=date(2025, 4, 1))

    dates = kept.drop_duplicates("game_id").set_index("game_id")["game_date"]
    assert dates.dt.strftime("%Y-%m-%d").to_dict() == {
        _game_id(2025, 1): "2025-01-09",
        _game_id(2025, 2): "2025-02-19",
    }
    assert counts == {
        "dropped_undated_games": 1,
        "dropped_unplayed_games": 2,
        "dropped_future_games": 1,
    }


def test_an_unplayed_past_game_is_cancelled_once_its_season_is_over():
    schedule = _schedule_frame(2013, [(1, "2013-04-16", False), (2, "2013-04-15", True)])

    unplayed = unplayed_games(schedule, 2013, current_end_year=2027, today=TODAY)

    assert unplayed[["game_id", "listed_date", "status"]].to_dict("records") == [
        {"game_id": _game_id(2013, 1), "listed_date": "2013-04-16", "status": "cancelled"}
    ]


def test_a_game_this_season_is_postponed_once_the_release_has_moved_past_it():
    """Last night's games have no result until the next rebuild; they are not postponed."""
    schedule = _schedule_frame(
        2027,
        [
            (1, "2026-11-01", False),  # skipped over: later games have results
            (2, "2026-11-02", True),
            (3, "2026-11-03", False),  # after the last result: not in yet
            (4, "2026-12-11", False),  # future
        ],
    )

    unplayed = unplayed_games(schedule, 2027, current_end_year=2027, today=TODAY)

    assert unplayed["game_id"].tolist() == [_game_id(2027, 1)]
    assert unplayed["status"].tolist() == ["postponed"]


def test_a_placeholder_without_teams_is_not_an_unplayed_game():
    schedule = _schedule_frame(2027, [(1, "2026-11-01", False)]).assign(home_team_id=0)

    assert unplayed_games(schedule, 2027, current_end_year=2027, today=TODAY).empty


def test_the_unplayed_record_keeps_first_sight_and_drops_games_since_played(tmp_path):
    postponed = {2027: [(1, "2026-11-02", True), (2, "2026-11-01", False)]}
    _fetch(tmp_path, _stub("t1", [2027], postponed), [2027])
    first = pd.read_csv(tmp_path / "unplayed.csv", dtype={"game_id": str})
    assert first[["game_id", "status"]].to_dict("records") == [
        {"game_id": _game_id(2027, 2), "status": "postponed"}
    ]

    _fetch(tmp_path, _stub("t2", [2027], postponed), [2027])
    again = pd.read_csv(tmp_path / "unplayed.csv", dtype={"game_id": str})
    assert again["recorded_at"].tolist() == first["recorded_at"].tolist()

    made_up = {2027: [(1, "2026-11-02", True), (2, "2026-11-03", True)]}
    _fetch(tmp_path, _stub("t3", [2027], made_up), [2027])
    assert pd.read_csv(tmp_path / "unplayed.csv").empty
