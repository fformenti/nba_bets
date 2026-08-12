"""Predictions accumulate across runs, so writing them must not duplicate rows.

The pipeline previously appended unconditionally, so re-predicting a slate —
routine after a retrain — silently doubled every game. Anything counting those
rows, above all the accuracy scorecard, would then double-count.
"""

import pandas as pd
import pytest

from src.ml.prediction.pipeline import upsert_predictions


def _predictions(game_ids, probability=0.7):
    return pd.DataFrame(
        [
            {
                "gameId": game_id,
                "gameDate": f"2026-02-{game_id:02d}",
                "prediction": 1,
                "home_win_probability": probability,
            }
            for game_id in game_ids
        ]
    )


@pytest.fixture
def output_path(tmp_path):
    return tmp_path / "predictions.csv"


def test_first_write_creates_the_file(output_path):
    assert upsert_predictions(_predictions([1, 2, 3]), output_path) == 3
    assert len(pd.read_csv(output_path)) == 3


def test_rerunning_the_same_slate_does_not_grow_the_file(output_path):
    upsert_predictions(_predictions([1, 2, 3]), output_path)
    assert upsert_predictions(_predictions([1, 2, 3]), output_path) == 3


def test_rerun_keeps_the_newer_prediction(output_path):
    upsert_predictions(_predictions([1], probability=0.60), output_path)
    upsert_predictions(_predictions([1], probability=0.85), output_path)

    written = pd.read_csv(output_path)
    assert len(written) == 1
    assert written.loc[0, "home_win_probability"] == 0.85


def test_new_games_are_appended(output_path):
    upsert_predictions(_predictions([1, 2]), output_path)
    assert upsert_predictions(_predictions([3, 4]), output_path) == 4


def test_rows_left_by_the_old_three_model_routing_are_collapsed(output_path):
    """A game is one row.

    Prediction files written under the per-conference split carry a row per
    model that scored the game, distinguished by a conference_filter column that
    is no longer written. Re-predicting such a game must leave one row, not a
    third — the betting path sizes one order plan per row.
    """
    stale = _predictions([1], probability=0.60)
    stale["conference_filter"] = "same"
    stale.to_csv(output_path, index=False)

    total = upsert_predictions(_predictions([1], probability=0.85), output_path)

    written = pd.read_csv(output_path)
    assert total == 1
    assert written.loc[0, "home_win_probability"] == 0.85


def test_output_is_sorted_by_date(output_path):
    upsert_predictions(_predictions([3]), output_path)
    upsert_predictions(_predictions([1]), output_path)

    written = pd.read_csv(output_path)
    assert written["gameDate"].is_monotonic_increasing


def test_parent_directory_is_created(tmp_path):
    nested = tmp_path / "does" / "not" / "exist" / "predictions.csv"
    assert upsert_predictions(_predictions([1]), nested) == 1
    assert nested.exists()


def test_a_file_carrying_the_stale_column_still_dedupes(output_path):
    """The key is gameId alone, so an extra column changes nothing."""
    legacy = _predictions([1, 2])
    legacy["conference_filter"] = "all"
    legacy.to_csv(output_path, index=False)

    total = upsert_predictions(legacy, output_path)
    assert total == 2


def test_rerunning_a_slate_leaves_the_file_byte_identical(output_path):
    """Re-running an unchanged slate must not rewrite the file differently.

    Every game in a slate shares a date, so sorting on the date alone left them
    all tied — and pandas' default quicksort orders ties arbitrarily, so the CSV
    churned on each run even when nothing about the predictions had changed.
    """
    slate = _predictions([1, 2, 3, 4, 5])
    slate["gameDate"] = "2026-02-09"

    upsert_predictions(slate, output_path)
    first = output_path.read_bytes()
    upsert_predictions(slate, output_path)

    assert output_path.read_bytes() == first
