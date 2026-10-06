"""Tests for playoff standings helpers (team-game expansion, schedule length)."""

import pandas as pd
import pytest

from src.etl.features.playoff_standings import (
    _build_team_game_rows,
    _merge_games_remaining,
    _compute_cumulative_stats,
)


@pytest.fixture
def balanced_mini_schedule():
    """Four teams, six games, one season — three games per team (balanced)."""
    return pd.DataFrame(
        {
            "gameId": ["g1", "g2", "g3", "g4", "g5", "g6"],
            "gameDate": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-03",
                    "2025-01-04",
                ]
            ),
            "gameDateOnlyStr": [
                "2025-01-01",
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
                "2025-01-03",
                "2025-01-04",
            ],
            "season": ["2024/25"] * 6,
            "hometeamId": [1, 3, 2, 1, 3, 1],
            "awayteamId": [2, 4, 4, 2, 4, 3],
            "winner": [1, 3, 2, 2, 4, 1],
            "homeScore": [100, 90, 90, 80, 100, 110],
            "awayScore": [90, 80, 100, 100, 90, 90],
            "hometeamConference": ["East", "East", "East", "East", "East", "East"],
            "awayteamConference": ["East", "East", "East", "East", "East", "East"],
        }
    )


def test_build_team_game_rows_equal_counts_per_team(balanced_mini_schedule):
    team_games = _build_team_game_rows(balanced_mini_schedule)
    counts = team_games.groupby(["teamId", "season"]).size()
    assert counts.nunique() == 1
    assert counts.iloc[0] == 3


def test_build_team_game_rows_deduplicates_duplicate_game_rows(balanced_mini_schedule):
    doubled = pd.concat([balanced_mini_schedule, balanced_mini_schedule], ignore_index=True)
    team_games = _build_team_game_rows(doubled)
    counts = team_games.groupby(["teamId", "season"]).size()
    assert counts.nunique() == 1
    assert counts.iloc[0] == 3


def test_merge_games_remaining_uses_season_schedule_length(balanced_mini_schedule):
    team_games = _compute_cumulative_stats(_build_team_game_rows(balanced_mini_schedule))
    merged = _merge_games_remaining(team_games)
    # 6 unique games, 4 teams -> 2 * 6 // 4 == 3 games per team (league length)
    schedule_len = 3
    assert (merged["games_played"] + merged["games_remaining"] == schedule_len).all()
    assert merged["games_remaining"].notna().all()


def test_merge_games_remaining_never_goes_negative(balanced_mini_schedule):
    """An unbalanced season must not hand a team negative games left.

    ``total_season_games`` is one league-wide number, so any team that plays
    more than the league average gets a negative remainder. The 2019/20 bubble
    is the real case — 22 teams played ~72 games against an average of ~66 — and
    a negative remainder puts ``max_wins`` *below* current wins inside
    ``_rank_conference_snapshot``, which is the only way a team can come out
    flagged as both clinched and eliminated. Seven rows did.
    """
    # Team 1 plays a seventh game after the others are done: 4 games against a
    # league length of 3.
    extra = pd.DataFrame(
        {
            "gameId": ["g7"],
            "gameDate": pd.to_datetime(["2025-01-05"]),
            "gameDateOnlyStr": ["2025-01-05"],
            "season": ["2024/25"],
            "hometeamId": [1],
            "awayteamId": [4],
            "winner": [1],
            "homeScore": [101],
            "awayScore": [99],
            "hometeamConference": ["East"],
            "awayteamConference": ["East"],
        }
    )
    unbalanced = pd.concat([balanced_mini_schedule, extra], ignore_index=True)
    team_games = _compute_cumulative_stats(_build_team_game_rows(unbalanced))
    merged = _merge_games_remaining(team_games)

    assert (merged["games_remaining"] >= 0).all()
    assert (merged["post_games_remaining"] >= 0).all()
