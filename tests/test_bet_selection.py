"""One game, one bet.

The prediction file used to carry a row per (gameId, conference_filter), so a
game routed to its conference model *and* scored by the all-games model appeared
twice. `load_predictions_for_date` filtered only on date, so `build_daily_bets`
sized every game twice — two order plans against the same market, from two
different probabilities, each with a full GAME_BUDGET.

One model now predicts every game and the file is one row per game, so a
duplicate has no legitimate reading left. It is refused rather than resolved.
"""

import pandas as pd
import pytest

from src.betting import bets


def _predictions(rows):
    return pd.DataFrame(
        [
            {
                "gameId": game_id,
                "gameDateOnlyStr": date,
                "season": "2025/26",
                "hometeamId": 1,
                "awayteamId": 2,
                "hometeamName": "Home",
                "awayteamName": "Away",
                "home_win_probability": probability,
            }
            for game_id, date, probability in rows
        ]
    )


@pytest.fixture
def predictions_file(tmp_path, monkeypatch):
    """Point the loader at a temp file and stub out the Polymarket slug lookup."""
    path = tmp_path / "upcoming_games_predictions.csv"
    monkeypatch.setattr(bets, "UPCOMING_GAMES_PREDICTIONS_PATH", path)
    monkeypatch.setattr(
        bets, "get_game_slug", lambda away, home, date, season: f"{away}-{home}-{date}"
    )

    def write(rows):
        _predictions(rows).to_csv(path, index=False)
        return path

    return write


def test_one_row_per_game_is_the_normal_case(predictions_file):
    predictions_file(
        [
            (101, "2026-01-25", 0.60),
            (102, "2026-01-25", 0.71),
        ]
    )
    loaded = bets.load_predictions_for_date("2026-01-25")

    assert loaded["gameId"].tolist() == [101, 102]


def test_a_legacy_file_with_two_models_per_game_is_refused(predictions_file):
    """Rows left by the three-model routing are no longer disambiguable.

    Re-running `make predict-upcoming` collapses them; guessing which of two
    disagreeing probabilities to bet is not this function's call to make.
    """
    predictions_file(
        [
            (101, "2026-01-25", 0.60),
            (101, "2026-01-25", 0.51),
            (102, "2026-01-25", 0.71),
            (102, "2026-01-25", 0.73),
        ]
    )
    with pytest.raises(ValueError, match=r"more than one prediction: \[101, 102\]"):
        bets.load_predictions_for_date("2026-01-25")


def test_other_dates_are_excluded(predictions_file):
    predictions_file(
        [
            (101, "2026-01-25", 0.60),
            (201, "2026-01-26", 0.55),
        ]
    )
    assert bets.load_predictions_for_date("2026-01-26")["gameId"].tolist() == [201]


def test_an_empty_slate_returns_the_expected_columns(predictions_file):
    predictions_file([(101, "2026-01-25", 0.60)])
    loaded = bets.load_predictions_for_date("2026-01-27")

    assert loaded.empty
    assert list(loaded.columns) == bets.BET_COLUMNS


def test_two_rows_for_one_game_is_an_error(predictions_file):
    """A duplicate must never be sized twice, whatever produced it."""
    predictions_file(
        [
            (101, "2026-01-25", 0.60),
            (101, "2026-01-25", 0.64),
        ]
    )
    with pytest.raises(ValueError, match="more than one"):
        bets.load_predictions_for_date("2026-01-25")


def test_a_stale_file_carrying_conference_filter_still_loads(predictions_file):
    """The column is no longer written, but an old file must not break loading."""
    frame = _predictions([(101, "2026-01-25", 0.60)])
    frame["conference_filter"] = "all"
    frame.to_csv(bets.UPCOMING_GAMES_PREDICTIONS_PATH, index=False)

    assert len(bets.load_predictions_for_date("2026-01-25")) == 1
