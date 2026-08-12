"""The ETL and prediction must not share one feature-table directory.

Both jobs build the same 31 intermediate CSVs under the same filenames, but from
very different frames: `make build-features` uses all of history (~132K team
rows), `make predict-upcoming` uses one season plus the slate (~1.7K). While they
shared a directory, every prediction run silently truncated the ETL's tables —
invisibly, because the artifact everything downstream reads
(`games_features.csv`) is written only by the ETL and looked fine.

These tests pin the two properties that fix it: prediction writes and reads
somewhere else, and a build never inherits a table it did not just write.
"""

import inspect

import pandas as pd
import pytest

from src.config.paths import PREDICTION_FEATURES_DIR, REGULAR_SEASON_FEATURES_DIR
from src.etl.features import aggregator
from src.ml.prediction import features as prediction_features


def test_prediction_features_dir_is_not_the_etl_dir():
    """The whole point: two jobs, two directories, same filenames."""
    assert PREDICTION_FEATURES_DIR != REGULAR_SEASON_FEATURES_DIR
    assert REGULAR_SEASON_FEATURES_DIR not in PREDICTION_FEATURES_DIR.parents


def test_prediction_builds_and_reads_its_own_directory(monkeypatch):
    """Prediction must never write the ETL's tables, nor read them back."""
    seen = {}

    def fake_create(games, features_config, output_dir=None):
        seen["written_to"] = output_dir

    def fake_merge(games, features_dir=None):
        seen["read_from"] = features_dir
        return games.copy()

    monkeypatch.setattr(prediction_features, "create_features_tables_from_config", fake_create)
    monkeypatch.setattr(prediction_features, "merge_features", fake_merge)

    upcoming = pd.DataFrame({"gameId": [1, 2], "season": ["2025/26"] * 2})
    history = pd.DataFrame({"gameId": [3, 4], "season": ["2025/26"] * 2})

    prediction_features.build_prediction_feature_base(
        upcoming_games=upcoming,
        historical_features=history,
        features_config=object(),
    )

    assert seen["written_to"] == PREDICTION_FEATURES_DIR
    assert seen["read_from"] == PREDICTION_FEATURES_DIR


def test_already_played_games_are_dropped_from_the_slate(monkeypatch):
    """A slate game already in history keeps its historical row, not the 0-0 placeholder.

    `fetch-upcoming-games` writes the slate before tip-off and never clears it,
    so it overlaps `append-game-results` by design. Keeping both rows duplicated
    the gameId into every feature table, and `merge_features` doubled the frame
    at each of its ~24 joins — a 10-game slate reached 7.3M rows and the process
    was OOM-killed.
    """
    built_from = {}

    def fake_create(games, features_config, output_dir=None):
        built_from["games"] = games.copy()

    monkeypatch.setattr(prediction_features, "create_features_tables_from_config", fake_create)
    monkeypatch.setattr(
        prediction_features, "merge_features", lambda games, features_dir=None: games.copy()
    )

    # gameId 2 was played already: history has the real score, the slate a placeholder.
    history = pd.DataFrame({"gameId": [1, 2], "homeScore": [100, 110]})
    upcoming = pd.DataFrame({"gameId": [2, 3], "homeScore": [0, 0]})

    prediction_features.build_prediction_feature_base(
        upcoming_games=upcoming,
        historical_features=history,
        features_config=object(),
    )

    combined = built_from["games"]
    assert combined["gameId"].is_unique
    assert combined.loc[combined["gameId"] == 2, "homeScore"].item() == 110


def test_build_clears_tables_left_by_a_previous_run(tmp_path, monkeypatch):
    """`merge_features` reads optional tables behind `if path.exists()`.

    Without a clean-out, a table this config no longer builds would still be
    merged, carrying another run's inputs into this one's features.
    """
    stale = tmp_path / "teams_records.csv"
    stale.write_text("gameId,season,teamId\n1,2025/26,1610612741\n")
    keep = tmp_path / "notes.txt"
    keep.write_text("not a feature table")

    # Stop right after the clean-out; the builders themselves are not under test.
    class Stop(Exception):
        pass

    def boom(*args, **kwargs):
        raise Stop

    monkeypatch.setattr(aggregator, "calculate_home_record", boom)

    with pytest.raises(Stop):
        aggregator.create_features_tables(pd.DataFrame(), output_dir=tmp_path)

    assert not stale.exists()
    assert keep.exists(), "only *.csv is cleared"


def test_etl_directory_is_the_default():
    """`make build-features` passes no directory and must keep getting its own."""
    create = inspect.signature(aggregator.create_features_tables)
    merge = inspect.signature(aggregator.merge_features)

    assert create.parameters["output_dir"].default is None
    assert merge.parameters["features_dir"].default is None
