"""Scoring emitted predictions against games that were actually played.

Two things here are easy to get silently wrong and expensive if missed:

1. ``winner`` is the winning *teamId*, while ``prediction`` is a 0/1 home-win
   flag. Comparing them directly yields a plausible-looking but badly wrong
   accuracy, so the conversion is pinned explicitly.
2. Scoring must be idempotent. It reads the durable history rather than the
   transient results directory precisely so that re-running never changes the
   answer.
"""

import json

import pandas as pd
import pytest

from src.monitoring.scoring import join_predictions_to_outcomes, load_outcomes, score_predictions

HOME_A, AWAY_A = 1610612741, 1610612748
HOME_B, AWAY_B = 1610612752, 1610612755


@pytest.fixture
def history_path(tmp_path):
    """Two played games: game 1 a home win, game 2 an away win."""
    path = tmp_path / "games_updated_history.csv"
    pd.DataFrame(
        [
            {"gameId": 1, "hometeamId": HOME_A, "awayteamId": AWAY_A, "winner": HOME_A},
            {"gameId": 2, "hometeamId": HOME_B, "awayteamId": AWAY_B, "winner": AWAY_B},
        ]
    ).to_csv(path, index=False)
    return path


@pytest.fixture
def predictions_path(tmp_path):
    """Game 1 predicted correctly, game 2 wrongly, game 3 not yet played."""
    path = tmp_path / "upcoming_games_predictions.csv"
    pd.DataFrame(
        [
            {
                "gameId": 1, "hometeamId": HOME_A, "awayteamId": AWAY_A,
                "gameDateOnlyStr": "2026-02-01", "season": "2025/26",
                "conference_filter": "all", "prediction": 1,
                "home_win_probability": 0.80, "winner": 0,
            },
            {
                "gameId": 2, "hometeamId": HOME_B, "awayteamId": AWAY_B,
                "gameDateOnlyStr": "2026-02-02", "season": "2025/26",
                "conference_filter": "all", "prediction": 1,
                "home_win_probability": 0.65, "winner": 0,
            },
            {
                "gameId": 3, "hometeamId": HOME_A, "awayteamId": AWAY_B,
                "gameDateOnlyStr": "2026-02-03", "season": "2025/26",
                "conference_filter": "all", "prediction": 0,
                "home_win_probability": 0.40, "winner": 0,
            },
        ]
    ).to_csv(path, index=False)
    return path


@pytest.fixture
def empty_results_dir(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    return d


def test_winner_team_id_becomes_a_home_win_flag(history_path, empty_results_dir):
    outcomes = load_outcomes(history_path, empty_results_dir)

    assert outcomes.set_index("gameId")["actual_home_win"].to_dict() == {1: 1, 2: 0}


def test_pending_results_top_up_history(history_path, empty_results_dir):
    """A game fetched but not yet appended is still scoreable."""
    (empty_results_dir / "3.json").write_text(
        json.dumps({"gameId": 3, "hometeamId": HOME_A, "awayteamId": AWAY_B, "winner": HOME_A})
    )
    outcomes = load_outcomes(history_path, empty_results_dir)

    assert set(outcomes["gameId"]) == {1, 2, 3}
    assert outcomes.set_index("gameId").loc[3, "actual_home_win"] == 1


def test_history_wins_when_sources_disagree(history_path, empty_results_dir):
    """History is the settled record; a stale JSON must not override it."""
    (empty_results_dir / "1.json").write_text(
        json.dumps({"gameId": 1, "hometeamId": HOME_A, "awayteamId": AWAY_A, "winner": AWAY_A})
    )
    outcomes = load_outcomes(history_path, empty_results_dir)

    assert outcomes.set_index("gameId").loc[1, "actual_home_win"] == 1


def test_unplayed_predictions_are_excluded(predictions_path, history_path, empty_results_dir):
    predictions = pd.read_csv(predictions_path)
    scored = join_predictions_to_outcomes(
        predictions, load_outcomes(history_path, empty_results_dir)
    )

    assert set(scored["gameId"]) == {1, 2}, "game 3 has no outcome yet"
    assert scored.set_index("gameId")["correct"].to_dict() == {1: 1, 2: 0}


def test_scorecard_reports_accuracy(predictions_path, history_path, empty_results_dir, tmp_path):
    scorecard = score_predictions(
        predictions_path=predictions_path,
        historical_path=history_path,
        results_dir=empty_results_dir,
        scorecard_path=tmp_path / "scorecard.csv",
        scored_games_path=tmp_path / "scored.csv",
    )

    overall = scorecard[scorecard["scope"] == "overall"].iloc[0]
    assert overall["n_games"] == 2
    assert overall["accuracy"] == 0.5  # one right, one wrong
    assert (tmp_path / "scorecard.csv").exists()
    assert (tmp_path / "scored.csv").exists()


def test_scoring_is_idempotent(predictions_path, history_path, empty_results_dir, tmp_path):
    """Re-running must reproduce identical numbers — the reason history is the source."""
    kwargs = dict(
        predictions_path=predictions_path,
        historical_path=history_path,
        results_dir=empty_results_dir,
        scorecard_path=tmp_path / "scorecard.csv",
        scored_games_path=tmp_path / "scored.csv",
    )
    first = score_predictions(**kwargs)
    second = score_predictions(**kwargs)

    pd.testing.assert_frame_equal(first, second)


def test_no_predictions_file_is_a_clear_error(tmp_path, history_path, empty_results_dir):
    with pytest.raises(FileNotFoundError, match="predict-upcoming"):
        score_predictions(
            predictions_path=tmp_path / "missing.csv",
            historical_path=history_path,
            results_dir=empty_results_dir,
        )
