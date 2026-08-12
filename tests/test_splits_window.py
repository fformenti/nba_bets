"""The training window must reach recent basketball.

These tests exist because of a real bug: with proportional splitting
(``test_size``/``val_size`` of 0.2 each) over 46 seasons, the validation slice
swallowed 2014–2022 and the models never trained on a game newer than April
2014. Games per season roughly quintupled since 1950, so a fixed *fraction* of
the rows is a far longer span at the old end of the history than the recent end.

:func:`src.ml.datasets.splitters.season_split` replaces the fractions with
season boundaries. It runs on a synthetic table, so no real data is required.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml.datasets.splitters import season_split

SEASONS = [f"{y}/{str(y + 1)[-2:]}" for y in range(2015, 2025)]
GAMES_PER_SEASON = 20


@pytest.fixture
def frame():
    """X, y and a season Series, index-aligned, spanning 2015/16 → 2024/25."""
    seasons = [s for s in SEASONS for _ in range(GAMES_PER_SEASON)]
    n = len(seasons)
    rng = np.random.default_rng(0)

    X = pd.DataFrame({"feature": rng.random(n)})
    y = pd.Series(rng.integers(0, 2, n), name="winner")
    return X, y, pd.Series(seasons, name="season")


def test_test_set_is_every_season_from_the_boundary_on(frame):
    X, y, seasons = frame
    _, _, X_test, _, _, _ = season_split(
        X, y, seasons=seasons, test_start_season="2022/23", val_seasons=2
    )

    assert set(seasons[X_test.index]) == {"2022/23", "2023/24", "2024/25"}


def test_validation_is_the_n_seasons_before_the_boundary(frame):
    X, y, seasons = frame
    X_train, X_val, _, _, _, _ = season_split(
        X, y, seasons=seasons, test_start_season="2022/23", val_seasons=2
    )

    assert set(seasons[X_val.index]) == {"2020/21", "2021/22"}
    # The regression this whole change is about: train must run right up to the
    # validation window, not stop several seasons short of it.
    assert max(seasons[X_train.index]) == "2019/20"
    assert min(seasons[X_train.index]) == "2015/16"


def test_splits_partition_the_data(frame):
    X, y, seasons = frame
    X_train, X_val, X_test, y_train, y_val, y_test = season_split(
        X, y, seasons=seasons, test_start_season="2022/23", val_seasons=2
    )

    train, val, test = (set(f.index) for f in (X_train, X_val, X_test))
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(X.index)

    for features, target in ((X_train, y_train), (X_val, y_val), (X_test, y_test)):
        assert list(features.index) == list(target.index)


def test_a_season_after_the_boundary_is_test_not_validation(frame):
    """The straggler case that the frozen holdout got wrong.

    Games played after the holdout CSV was frozen sat on the test side of the
    boundary but were absent from the file, so they leaked into validation.
    A season comparison cannot miss them.
    """
    X, y, seasons = frame
    _, X_val, X_test, _, _, _ = season_split(
        X, y, seasons=seasons, test_start_season="2022/23", val_seasons=2
    )

    assert "2024/25" not in set(seasons[X_val.index])
    assert "2024/25" in set(seasons[X_test.index])


def test_boundary_past_the_end_of_the_data_raises(frame):
    X, y, seasons = frame
    with pytest.raises(ValueError, match="No games on or after season"):
        season_split(X, y, seasons=seasons, test_start_season="2030/31", val_seasons=2)


def test_too_few_seasons_for_the_validation_window_raises(frame):
    """Silently emptying the train set would be far worse than failing."""
    X, y, seasons = frame
    with pytest.raises(ValueError, match="Need at least"):
        season_split(X, y, seasons=seasons, test_start_season="2016/17", val_seasons=2)


def test_val_seasons_must_be_positive(frame):
    X, y, seasons = frame
    with pytest.raises(ValueError, match="val_seasons must be at least 1"):
        season_split(X, y, seasons=seasons, test_start_season="2022/23", val_seasons=0)


def test_seasons_must_cover_every_row(frame):
    X, y, seasons = frame
    with pytest.raises(ValueError, match="must cover every row"):
        season_split(
            X, y, seasons=seasons.iloc[:-5], test_start_season="2022/23", val_seasons=2
        )
