"""Era scaling of point margins: the day-level league scale and the LLM target.

margin_std is the std of the last MARGIN_STD_WINDOW played games through the
last earlier date with a game, across seasons. Every game on a date shares it,
and none sees another game from its own day. The LLM completion is the margin
divided by margin_std/reference, and evaluation multiplies the same factor back in.
"""

import numpy as np
import pandas as pd
import pytest

from src.config.constants import INTERMEDIATE_COLUMNS
from src.etl.features import point_differential
from src.etl.features.point_differential import (
    margin_std_by_date,
    _scale_margin,
    calculate_margin,
)
from src.etl.utils.common import add_game_time
from src.ml.llm.target import SPECS
from src.ml.llm.testers import ground_truth, margin_scale

HOME, AWAY = 1, 2


def _game(game_id, season, tip_off, home_score, away_score, played=True):
    return {
        "gameId": game_id,
        "season": season,
        "gameDate": pd.Timestamp(tip_off),
        "hometeamId": HOME,
        "awayteamId": AWAY,
        "homeScore": home_score,
        "awayScore": away_score,
        "winner": (HOME if home_score > away_score else AWAY) if played else 0,
    }


def _frame(rows):
    games = pd.DataFrame(rows)
    games["gameDateOnlyStr"] = games["gameDate"].dt.strftime("%Y-%m-%d")
    return add_game_time(games).set_index("gameId", drop=False)


@pytest.fixture
def one_day():
    # Opening night: one 19:00 game, two at 20:00, one at 22:00.
    return _frame(
        [
            _game(1, "2024/25", "2024-10-22 19:00", 110, 100),
            _game(2, "2024/25", "2024-10-22 20:00", 105, 100),
            _game(3, "2024/25", "2024-10-22 20:00", 100, 120),
            _game(4, "2024/25", "2024-10-22 22:00", 99, 98),
        ]
    )


def test_game_time_is_the_tip_off_wall_clock():
    games = _frame([_game(1, "2024/25", "2024-10-22 19:30", 100, 90)])
    assert games["gameTime"].tolist() == ["19:30"]


def _scale_of(games):
    """margin_std per gameId, joined by date as the ETL does."""
    table = margin_std_by_date(games).set_index("date")
    return games["gameDateOnlyStr"].map(table["margin_std"])


def test_games_on_one_date_share_a_scale_whatever_their_tip_off(one_day):
    next_day = _frame(
        list(one_day.drop(columns=["gameDateOnlyStr", "gameTime"]).to_dict("records"))
        + [
            _game(5, "2024/25", "2024-10-23 17:00", 120, 100),
            _game(6, "2024/25", "2024-10-23 22:00", 100, 101),
        ]
    )
    scale = _scale_of(next_day)

    # Opening night has no earlier day; the 22:00 game never sees the 17:00 one.
    assert scale.loc[[1, 2, 3, 4]].isna().all()
    assert scale.loc[5] == scale.loc[6] == pytest.approx(pd.Series([10, 5, -20, 1]).std())


def test_row_order_does_not_change_the_result(one_day):
    shuffled = one_day.sample(frac=1, random_state=0)
    pd.testing.assert_frame_equal(margin_std_by_date(one_day), margin_std_by_date(shuffled))


def test_a_placeholder_receives_a_value_but_contributes_nothing(one_day):
    slate = _frame(
        list(one_day.drop(columns=["gameDateOnlyStr", "gameTime"]).to_dict("records"))
        + [
            _game(5, "2024/25", "2024-10-23 19:00", 0, 0, played=False),
            _game(6, "2024/25", "2024-10-24 19:00", 0, 0, played=False),
        ]
    )
    table = margin_std_by_date(slate).set_index("date")
    opening_night = pd.Series([10, 5, -20, 1]).std()
    # Neither placeholder day counts as a day with a game.
    assert table.loc["2024-10-23", "last_date_with_game"] == "2024-10-22"
    assert table.loc["2024-10-24", "last_date_with_game"] == "2024-10-22"
    assert table.loc["2024-10-24", "margin_std"] == pytest.approx(opening_night)


def _season(season, start, margins, first_id):
    day = pd.Timestamp(start)
    return [
        _game(first_id + i, season, day + pd.Timedelta(days=i, hours=19), 100 + m, 100)
        for i, m in enumerate(margins)
    ]


def test_margin_std_is_the_std_of_the_trailing_window(monkeypatch):
    monkeypatch.setattr(point_differential, "MARGIN_STD_WINDOW", 150)
    margins = np.random.default_rng(0).integers(-25, 26, size=400)
    games = _frame(_season("2024/25", "2024-10-22", margins, 1))

    # One game a day: every earlier game until there are 150, then the last 150.
    expected = pd.Series(margins, dtype=float).rolling(150, min_periods=2).std().shift(1)
    assert np.allclose(_scale_of(games).to_numpy(), expected.to_numpy(), equal_nan=True)


def test_the_window_crosses_the_season_boundary_and_skips_off_days(monkeypatch):
    monkeypatch.setattr(point_differential, "MARGIN_STD_WINDOW", 120)
    last = np.random.default_rng(1).choice([-10, 10], size=200)
    games = _frame(
        _season("2023/24", "2023-10-24", last, 1)
        + _season("2024/25", "2024-10-22", [5, 5], 1001)
    )
    table = margin_std_by_date(games).set_index("date")

    # Opening night looks back across the off-season to the previous season's
    # final day, and is scaled by that season's last 120 games.
    final_day = games.loc[200, "gameDateOnlyStr"]
    assert table.loc["2024-10-22", "last_date_with_game"] == final_day
    assert table.loc["2024-10-22", "margin_std"] == pytest.approx(pd.Series(last[-120:]).std())


def test_the_opening_games_of_history_never_read_later_results():
    games = _frame(_season("1946/47", "1946-11-01", [10, -10, 30, 99], 1))
    scale = _scale_of(games)

    assert scale.loc[[1, 2]].isna().all()  # fewer than two earlier games
    assert scale.loc[3] == pytest.approx(pd.Series([10, -10]).std())
    assert scale.loc[4] == pytest.approx(pd.Series([10, -10, 30]).std())


def test_margin_scaled_is_in_units_of_the_margin_std():
    games = pd.DataFrame(
        {
            "gameId": [1, 2, 3],
            "gameDate": pd.to_datetime(["2024-10-22", "2024-10-24", "2024-10-26"]),
            "season": "2024/25",
            "teamId": HOME,
            "margin": [12.0, -6.0, 3.0],
            "margin_std": [12.0, 15.0, 15.0],
        }
    )
    games["margin_scaled"] = _scale_margin(games["margin"], games["margin_std"])
    scaled = calculate_margin(games, lags=[2], column="margin_scaled").set_index("gameId")
    # Game 3 averages the two previous games: (12/12 + -6/15) / 2.
    assert scaled.loc[3, "margin_scaled_avg_L2"] == pytest.approx((1.0 - 0.4) / 2)


def test_scaling_round_trips_through_evaluation():
    margin, season_std, reference = -9.0, 15.6, 13.0
    scale = season_std / reference
    completion = SPECS["regression"].completion(margin, scale)
    row = {"completion": completion, "margin": margin, "margin_scale": scale}

    assert completion == "-8"
    assert ground_truth(row) == margin
    # A prediction equal to the completion comes back within rounding of truth.
    assert abs(int(completion) * margin_scale(row) - margin) <= scale / 2


def test_an_unscaled_dataset_reads_as_before():
    row = {"completion": "+7"}
    assert ground_truth(row) == 7.0
    assert margin_scale(row) == 1.0


def test_neither_column_reaches_a_model():
    assert {"gameTime", "margin_std"} <= set(INTERMEDIATE_COLUMNS)
