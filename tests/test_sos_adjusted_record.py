"""Tests for SOS-Adjusted Record feature calculation.

Includes a worked example demonstrating two teams with the same raw
record but different SOS values, showing how multiplicative adjustment
changes their relative ranking.
"""

import numpy as np
import pandas as pd
import pytest

from src.etl.features.strength_of_schedule import calculate_strength_of_schedule
from src.etl.features.winning_percentage import (
    calculate_away_record,
    calculate_home_record,
    calculate_record,
)
from src.etl.features.sos_adjusted_record import calculate_sos_adjusted_record


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_records(games, record_lags):
    """Replicate aggregator's record-building flow."""
    home = calculate_home_record(games, lags=[])
    away = calculate_away_record(games, lags=[])
    return calculate_record(
        pd.concat([home, away], ignore_index=True), lags=record_lags
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def worked_example_games():
    """
    6 games, 4 teams, 1 season.

    Designed so that Team 1 and Team 3 arrive at the evaluation game (g6)
    with identical 1-1 records but different strength of schedule.

    Game results:
      g1  Jan 1: 1(H) vs 2(A) → 1 wins   | 1=1-0, 2=0-1
      g2  Jan 1: 3(H) vs 4(A) → 3 wins   | 3=1-0, 4=0-1
      g3  Jan 2: 2(H) vs 4(A) → 2 wins   | 2=1-1, 4=0-2
      g4  Jan 3: 1(H) vs 2(A) → 2 wins   | 1=1-1, 2=2-1
      g5  Jan 3: 3(H) vs 4(A) → 4 wins   | 3=1-1, 4=1-2
      g6  Jan 4: 1(H) vs 3(A) → 1 wins   | 1=2-1, 3=1-2

    At g6 (before the game happens):
      Team 1: 1-1 raw, opponents = [Team 2, Team 2]
        Team 2's cum_win_pct as of Jan 4 = 2/3 (2W-1L after g3,g4)
        → SOS = 2/3

      Team 3: 1-1 raw, opponents = [Team 4, Team 4]
        Team 4's cum_win_pct as of Jan 4 = 1/3 (1W-2L after g2,g3,g5)
        → SOS = 1/3

    league_avg_sos on Jan 4 = mean(2/3, 1/3) = 0.5

    SOS-adjusted (alpha=1.0):
      Team 1: 0.5 * (2/3 / 0.5) = 0.5 * 4/3 = 2/3 ≈ 0.667
      Team 3: 0.5 * (1/3 / 0.5) = 0.5 * 2/3 = 1/3 ≈ 0.333

    Same raw record → different adjusted records.
    """
    return pd.DataFrame(
        {
            "gameId": ["g1", "g2", "g3", "g4", "g5", "g6"],
            "gameDate": pd.to_datetime([
                "2025-01-01",
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
                "2025-01-03",
                "2025-01-04",
            ]),
            "season": ["2024/25"] * 6,
            "hometeamId": [1, 3, 2, 1, 3, 1],
            "awayteamId": [2, 4, 4, 2, 4, 3],
            "winner": [1, 3, 2, 2, 4, 1],
            "homeScore": [100, 100, 100, 90, 90, 100],
            "awayScore": [90, 90, 90, 100, 100, 90],
        }
    )


@pytest.fixture
def sample_games():
    """Same 4-team, 8-game fixture from test_strength_of_schedule.py."""
    return pd.DataFrame(
        {
            "gameId": [f"g{i}" for i in range(1, 9)],
            "gameDate": pd.to_datetime([
                "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04",
                "2025-01-05", "2025-01-06", "2025-01-07", "2025-01-08",
            ]),
            "season": ["2024/25"] * 8,
            "hometeamId": [1, 3, 1, 2, 4, 2, 1, 3],
            "awayteamId": [2, 4, 3, 4, 1, 3, 4, 2],
            "winner": [1, 3, 1, 2, 1, 3, 1, 3],
            "homeScore": [100, 100, 100, 100, 90, 90, 100, 100],
            "awayScore": [90, 90, 90, 90, 100, 100, 90, 90],
        }
    )


# ---------------------------------------------------------------------------
# Worked example: same record, different SOS
# ---------------------------------------------------------------------------

class TestWorkedExample:
    """Two teams with identical 1-1 records but different SOS."""

    def test_same_raw_record_different_adjusted(self, worked_example_games):
        """Core demonstration: Team 1 (tough schedule) ranks above Team 3 (easy schedule)."""
        records_df = _build_records(worked_example_games, record_lags=[10])
        sos_df = calculate_strength_of_schedule(
            worked_example_games, lags=[10], min_opponents=2,
        )

        result = calculate_sos_adjusted_record(
            records_df, sos_df, record_lags=[10], sos_lags=[10], alpha=1.0,
        )

        team1_g6 = result[(result["teamId"] == 1) & (result["gameId"] == "g6")]
        team3_g6 = result[(result["teamId"] == 3) & (result["gameId"] == "g6")]

        # Both have raw record_L10 = 0.5, but adjusted values differ
        adj_team1 = team1_g6["sos_adj_record_L10"].iloc[0]
        adj_team3 = team3_g6["sos_adj_record_L10"].iloc[0]

        assert abs(adj_team1 - 2 / 3) < 1e-6, f"Team 1 adj={adj_team1}, expected 2/3"
        assert abs(adj_team3 - 1 / 3) < 1e-6, f"Team 3 adj={adj_team3}, expected 1/3"

        # Team 1 ranks higher than Team 3 after adjustment
        assert adj_team1 > adj_team3


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------

class TestFunctional:

    def test_sos_adj_record_team3_g8(self, sample_games):
        """
        Team 3 at g8 (Jan 8).

        record_L82 = round(mean([1, 0, 1]), 2) = 0.67
        sos_L82 = (0 + 1.0 + 1/3) / 3 = 4/9
        league_avg_sos at Jan 8 = mean(4/9, 5/9) = 0.5
        sos_adj_record_L82 = 0.67 * (4/9 / 0.5) = 0.67 * 8/9
        """
        records_df = _build_records(sample_games, record_lags=[82])
        sos_df = calculate_strength_of_schedule(sample_games, lags=[82])

        result = calculate_sos_adjusted_record(
            records_df, sos_df, record_lags=[82], sos_lags=[82], alpha=1.0,
        )

        row = result[(result["teamId"] == 3) & (result["gameId"] == "g8")]
        # record_L82 is rounded to 0.67, so expected = 0.67 * 8/9
        expected = 0.67 * (8 / 9)
        assert abs(row["sos_adj_record_L82"].iloc[0] - expected) < 1e-6

    def test_alpha_zero_equals_raw(self, worked_example_games):
        """With alpha=0, (sos / avg)^0 = 1, so adjusted == raw."""
        records_df = _build_records(worked_example_games, record_lags=[10])
        sos_df = calculate_strength_of_schedule(
            worked_example_games, lags=[10], min_opponents=2,
        )

        result = calculate_sos_adjusted_record(
            records_df, sos_df, record_lags=[10], sos_lags=[10], alpha=0.0,
        )

        team1_g6 = result[(result["teamId"] == 1) & (result["gameId"] == "g6")]
        assert abs(team1_g6["sos_adj_record_L10"].iloc[0] - 0.5) < 1e-6

    def test_nan_early_season(self, sample_games):
        """Early-season rows where SOS is NaN produce NaN adjusted values."""
        records_df = _build_records(sample_games, record_lags=[82])
        sos_df = calculate_strength_of_schedule(sample_games, lags=[82])

        result = calculate_sos_adjusted_record(
            records_df, sos_df, record_lags=[82], sos_lags=[82], alpha=1.0,
        )

        # Team 1's first game (g1): SOS is NaN → adjusted should be NaN
        row = result[(result["teamId"] == 1) & (result["gameId"] == "g1")]
        assert np.isnan(row["sos_adj_record_L82"].iloc[0])


# ---------------------------------------------------------------------------
# Column / lag matching tests
# ---------------------------------------------------------------------------

class TestLagMatching:

    def test_only_intersection_lags_produce_columns(self, sample_games):
        """record_lags=[1, 82] ∩ sos_lags=[10, 82] = {82}."""
        records_df = _build_records(sample_games, record_lags=[1, 82])
        sos_df = calculate_strength_of_schedule(sample_games, lags=[10, 82])

        result = calculate_sos_adjusted_record(
            records_df, sos_df, record_lags=[1, 82], sos_lags=[10, 82], alpha=1.0,
        )

        assert "sos_adj_record_L82" in result.columns
        assert "sos_adj_record_L1" not in result.columns
        assert "sos_adj_record_L10" not in result.columns

    def test_output_columns(self, worked_example_games):
        """Verify expected output column names."""
        records_df = _build_records(worked_example_games, record_lags=[10])
        sos_df = calculate_strength_of_schedule(
            worked_example_games, lags=[10], min_opponents=2,
        )

        result = calculate_sos_adjusted_record(
            records_df, sos_df, record_lags=[10], sos_lags=[10], alpha=1.0,
        )

        expected_cols = {"gameId", "season", "teamId", "gameDate",
                         "sos_adj_record_L10"}
        assert set(result.columns) == expected_cols

    def test_no_matched_lags_only_base_columns(self, sample_games):
        """No intersection → only join key columns, no adjusted columns."""
        records_df = _build_records(sample_games, record_lags=[1])
        sos_df = calculate_strength_of_schedule(sample_games, lags=[82])

        result = calculate_sos_adjusted_record(
            records_df, sos_df, record_lags=[1], sos_lags=[82], alpha=1.0,
        )

        assert set(result.columns) == {"gameId", "season", "teamId", "gameDate"}


# ---------------------------------------------------------------------------
# Season boundary test
# ---------------------------------------------------------------------------

def test_season_reset():
    """SOS-adjusted record resets at season boundaries."""
    games = pd.DataFrame(
        {
            "gameId": [f"g{i}" for i in range(1, 9)],
            "gameDate": pd.to_datetime([
                "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04",
                "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04",
            ]),
            "season": ["2023/24"] * 4 + ["2024/25"] * 4,
            "hometeamId": [1, 3, 1, 2, 1, 3, 1, 2],
            "awayteamId": [2, 4, 3, 4, 2, 4, 3, 4],
            "winner": [1, 3, 1, 2, 1, 3, 1, 2],
            "homeScore": [100] * 8,
            "awayScore": [90] * 8,
        }
    )

    records_df = _build_records(games, record_lags=[10])
    sos_df = calculate_strength_of_schedule(games, lags=[10], min_opponents=1)

    result = calculate_sos_adjusted_record(
        records_df, sos_df, record_lags=[10], sos_lags=[10], alpha=1.0,
    )

    # Team 1's first game in season 2 (g5): SOS is NaN (no prior opponents
    # in this season) → adjusted should be NaN
    row = result[(result["teamId"] == 1) & (result["gameId"] == "g5")]
    assert np.isnan(row["sos_adj_record_L10"].iloc[0])
