"""Predictions accumulate across runs, so writing them must not duplicate rows.

The pipeline previously appended unconditionally, so re-predicting a slate —
routine after a retrain — silently doubled every game. Anything counting those
rows, above all the accuracy scorecard, would then double-count.
"""

import pandas as pd
import pytest

from src.ml.prediction.pipeline import upsert_predictions


def _predictions(game_ids, probability=0.7, conference_filter="all"):
    return pd.DataFrame(
        [
            {
                "gameId": game_id,
                "gameDate": f"2026-02-{game_id:02d}",
                "conference_filter": conference_filter,
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


def test_same_game_under_different_conference_models_both_kept(output_path):
    """A game routed to two models is two legitimate rows, not a duplicate."""
    upsert_predictions(_predictions([1], conference_filter="all"), output_path)
    total = upsert_predictions(_predictions([1], conference_filter="same"), output_path)

    assert total == 2


def test_output_is_sorted_by_date(output_path):
    upsert_predictions(_predictions([3]), output_path)
    upsert_predictions(_predictions([1]), output_path)

    written = pd.read_csv(output_path)
    assert written["gameDate"].is_monotonic_increasing


def test_parent_directory_is_created(tmp_path):
    nested = tmp_path / "does" / "not" / "exist" / "predictions.csv"
    assert upsert_predictions(_predictions([1]), nested) == 1
    assert nested.exists()


def test_legacy_file_without_conference_filter_still_dedupes(output_path):
    """Files written before conference_filter existed must still be handled."""
    legacy = _predictions([1, 2]).drop(columns=["conference_filter"])
    legacy.to_csv(output_path, index=False)

    total = upsert_predictions(legacy, output_path)
    assert total == 2
