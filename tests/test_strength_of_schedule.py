"""Tests for Strength of Schedule feature calculation.

Worked example with 4 teams (ids 1-4), 8 games in one season.
Verifies SOS values match hand-computed expectations.
"""

import numpy as np
import pandas as pd
import pytest

from src.etl.features.strength_of_schedule import calculate_strength_of_schedule


@pytest.fixture
def sample_games():
    """
    4 teams, 8 games, single season.

    Game results:
      g1  Jan 1: 1(H) vs 2(A) → 1 wins   | Records after: 1=1-0, 2=0-1
      g2  Jan 2: 3(H) vs 4(A) → 3 wins   | Records after: 3=1-0, 4=0-1
      g3  Jan 3: 1(H) vs 3(A) → 1 wins   | Records after: 1=2-0, 3=1-1
      g4  Jan 4: 2(H) vs 4(A) → 2 wins   | Records after: 2=1-1, 4=0-2
      g5  Jan 5: 4(H) vs 1(A) → 1 wins   | Records after: 1=3-0, 4=0-3
      g6  Jan 6: 2(H) vs 3(A) → 3 wins   | Records after: 2=1-2, 3=2-1
      g7  Jan 7: 1(H) vs 4(A) → 1 wins   | Records after: 1=4-0, 4=0-4
      g8  Jan 8: 3(H) vs 2(A) → 3 wins   | Records after: 3=3-1, 2=1-3
    """
    return pd.DataFrame(
        {
            "gameId": ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8"],
            "gameDate": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-04",
                    "2025-01-05",
                    "2025-01-06",
                    "2025-01-07",
                    "2025-01-08",
                ]
            ),
            "season": ["2024/25"] * 8,
            "hometeamId": [1, 3, 1, 2, 4, 2, 1, 3],
            "awayteamId": [2, 4, 3, 4, 1, 3, 4, 2],
            "winner": [1, 3, 1, 2, 1, 3, 1, 3],
        }
    )


def test_sos_team1_game_g7(sample_games):
    """
    Team 1 at game g7 (Jan 7), window=10.

    Previous opponents: 2 (g1), 3 (g3), 4 (g5) → 3 opponents.
    Opponent records as of Jan 7:
      Team 2: games g1,g4,g6 → 1W-2L → 1/3
      Team 3: games g2,g3,g6 → 2W-1L → 2/3
      Team 4: games g2,g4,g5 → 0W-3L → 0/3
    SOS = (1/3 + 2/3 + 0) / 3 = 1/3.
    """
    result = calculate_strength_of_schedule(sample_games, lags=[10])
    row = result[(result["teamId"] == 1) & (result["gameId"] == "g7")]
    assert len(row) == 1
    assert abs(row["sos_L10"].iloc[0] - 1 / 3) < 1e-6


def test_sos_team3_game_g8(sample_games):
    """
    Team 3 at game g8 (Jan 8), window=10.

    Previous opponents: 4 (g2), 1 (g3), 2 (g6) → 3 opponents.
    Opponent records as of Jan 8:
      Team 4: games g2,g4,g5,g7 → 0W-4L → 0.0
      Team 1: games g1,g3,g5,g7 → 4W-0L → 1.0
      Team 2: games g1,g4,g6    → 1W-2L → 1/3
    SOS = (0.0 + 1.0 + 1/3) / 3 ≈ 0.444.
    """
    result = calculate_strength_of_schedule(sample_games, lags=[10])
    row = result[(result["teamId"] == 3) & (result["gameId"] == "g8")]
    expected = (0.0 + 1.0 + 1 / 3) / 3
    assert abs(row["sos_L10"].iloc[0] - expected) < 1e-6


def test_sos_nan_for_insufficient_games(sample_games):
    """First games with < 3 previous opponents should be NaN."""
    result = calculate_strength_of_schedule(sample_games, lags=[10])

    # Team 1's first game g1: 0 previous opponents → NaN
    val_g1 = result[(result["teamId"] == 1) & (result["gameId"] == "g1")][
        "sos_L10"
    ].iloc[0]
    assert np.isnan(val_g1)

    # Team 1's second game g3: 1 previous opponent → NaN
    val_g3 = result[(result["teamId"] == 1) & (result["gameId"] == "g3")][
        "sos_L10"
    ].iloc[0]
    assert np.isnan(val_g3)

    # Team 1's third game g5: 2 previous opponents → NaN
    val_g5 = result[(result["teamId"] == 1) & (result["gameId"] == "g5")][
        "sos_L10"
    ].iloc[0]
    assert np.isnan(val_g5)


def test_sos_team1_game_g5_exactly_at_threshold(sample_games):
    """
    Team 1 at game g5 (Jan 5), window=10, min_opponents=2.

    Previous opponents: 2 (g1), 3 (g3) → 2 opponents.
    Opponent records as of Jan 5:
      Team 2: games g1,g4 → 1W-1L → 0.5
      Team 3: games g2,g3 → 1W-1L → 0.5
    SOS = (0.5 + 0.5) / 2 = 0.5.
    """
    result = calculate_strength_of_schedule(
        sample_games, lags=[10], min_opponents=2
    )
    row = result[(result["teamId"] == 1) & (result["gameId"] == "g5")]
    assert abs(row["sos_L10"].iloc[0] - 0.5) < 1e-6


def test_multiple_lags(sample_games):
    """Multiple lag windows produce separate columns."""
    result = calculate_strength_of_schedule(sample_games, lags=[3, 10])
    assert "sos_L3" in result.columns
    assert "sos_L10" in result.columns


def test_small_window_limits_opponents(sample_games):
    """
    Team 1 at game g7 (Jan 7), window=2.

    Only looks at last 2 opponents: 3 (g3), 4 (g5).
    But min_opponents=3 → NaN (only 2 opponents in window).
    """
    result = calculate_strength_of_schedule(
        sample_games, lags=[2], min_opponents=3
    )
    row = result[(result["teamId"] == 1) & (result["gameId"] == "g7")]
    assert np.isnan(row["sos_L2"].iloc[0])


def test_small_window_with_low_min(sample_games):
    """
    Team 1 at game g7 (Jan 7), window=2, min_opponents=2.

    Last 2 opponents: 3 (g3), 4 (g5).
    Opponent records as of Jan 7:
      Team 3: 2W-1L → 2/3
      Team 4: 0W-3L → 0/3
    SOS = (2/3 + 0) / 2 = 1/3.
    """
    result = calculate_strength_of_schedule(
        sample_games, lags=[2], min_opponents=2
    )
    row = result[(result["teamId"] == 1) & (result["gameId"] == "g7")]
    assert abs(row["sos_L2"].iloc[0] - 1 / 3) < 1e-6


def test_empty_lags(sample_games):
    """Empty lags list should return base columns only."""
    result = calculate_strength_of_schedule(sample_games, lags=[])
    assert list(result.columns) == ["gameId", "season", "teamId", "gameDate"]


def test_output_row_count(sample_games):
    """8 games × 2 teams per game = 16 team-game rows."""
    result = calculate_strength_of_schedule(sample_games, lags=[10])
    assert len(result) == 16


def test_season_reset():
    """SOS resets at season boundaries — no carry-over."""
    games = pd.DataFrame(
        {
            "gameId": [f"g{i}" for i in range(1, 9)],
            "gameDate": pd.to_datetime(
                [
                    # Season 1: 4 games
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    # Season 2: 4 games (same teams, fresh start)
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-04",
                ]
            ),
            "season": ["2023/24"] * 4 + ["2024/25"] * 4,
            "hometeamId": [1, 3, 1, 2, 1, 3, 1, 2],
            "awayteamId": [2, 4, 3, 4, 2, 4, 3, 4],
            "winner": [1, 3, 1, 2, 1, 3, 1, 2],
        }
    )
    result = calculate_strength_of_schedule(games, lags=[10], min_opponents=1)

    # Team 1's first game in season 2 (g5) has 0 previous opponents → NaN
    row = result[(result["teamId"] == 1) & (result["gameId"] == "g5")]
    assert np.isnan(row["sos_L10"].iloc[0])
