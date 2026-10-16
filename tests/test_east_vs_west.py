"""Unit tests for the East-vs-West conference-matchup ETL builders.

Three cross-conference games, one same-conference game that must be excluded
entirely, across three dates of a single season:

    Day 1: East (home) vs West (away) -> East wins   (cross, hosted by East)
    Day 1: West (home) vs West (away) -> West wins   (same-conference, excluded)
    Day 2: West (home) vs East (away) -> West wins   (cross, hosted by West)
    Day 3: East (home) vs West (away) -> West wins   (cross, hosted by East)

Hand-computed expectations for the season as a whole:
    east_wins = 1 (day 1), west_wins = 2 (days 2, 3)
    games_played_at_east = 2 (days 1, 3), games_played_at_west = 1 (day 2)
    east_wins_at_east = 1 (day 1), east_wins_at_west = 0 (day 2)
"""

import pandas as pd
import pytest

from src.etl.features.east_vs_west import (
    make_east_west_record,
    make_last_season_east_west_record,
)


def _games() -> pd.DataFrame:
    dates = pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-03"])
    return pd.DataFrame(
        {
            "gameDate": dates,
            "gameDateOnlyStr": dates.strftime("%Y-%m-%d"),
            "season": ["2022/23"] * 4,
            "hometeamConference": ["East", "West", "West", "East"],
            "awayteamConference": ["West", "West", "East", "West"],
            "winnerteamConference": ["East", "West", "West", "West"],
        }
    )


def test_games_played_east_vs_west_is_exported_and_shifted():
    """The cumulative count is shifted a day, so a row never sees its own game."""
    out = make_east_west_record(_games())
    by_date = out.set_index("gameDateOnlyStr")["games_played_east_vs_west"]

    assert by_date["2023-01-01"] == 0  # no interconference game played yet
    assert by_date["2023-01-02"] == 1  # day 1's East-hosted game
    assert by_date["2023-01-03"] == 2  # day 1's + day 2's games


def test_last_season_record_adjusted_matches_hand_computation():
    out = make_last_season_east_west_record(_games())
    row = out.iloc[0]

    assert row["season"] == "2022/23"
    # east_wr_at_east = 1/2, east_wr_at_west = 0/1 -> mean = 0.25
    assert row["last_season_east_record_adjusted"] == pytest.approx(0.25)
    # west_wr_at_east = 1/2, west_wr_at_west = 1/1 -> mean = 0.75
    assert row["last_season_west_record_adjusted"] == pytest.approx(0.75)


def test_last_season_record_at_east_matches_hand_computation():
    out = make_last_season_east_west_record(_games(), location="East")
    assert out.iloc[0]["last_season_east_record_at_east"] == pytest.approx(0.5)


def test_last_season_record_at_west_matches_hand_computation():
    out = make_last_season_east_west_record(_games(), location="West")
    assert out.iloc[0]["last_season_west_record_at_west"] == pytest.approx(1.0)


def test_last_season_record_is_one_row_per_season_not_per_date():
    out = make_last_season_east_west_record(_games())
    assert len(out) == 1
    assert "gameDateOnlyStr" not in out.columns
