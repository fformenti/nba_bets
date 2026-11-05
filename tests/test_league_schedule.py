"""The schedule pull's failure modes.

daily-cycle runs this step first, so how it fails decides whether the results
step behind it runs at all.
"""

import pandas as pd
import pytest
import requests

from src.etl.collectors import league_schedule
from src.etl.collectors.league_schedule import fetch_league_schedule


@pytest.fixture
def unreachable(monkeypatch):
    def hang(*args, **kwargs):
        raise requests.exceptions.ReadTimeout("stats.nba.com read timed out")

    monkeypatch.setattr(league_schedule.scheduleleaguev2, "ScheduleLeagueV2", hang)


@pytest.fixture
def cached(tmp_path):
    path = tmp_path / "schedule.csv"
    pd.DataFrame({"gameId": [22500664], "gameDateTimeEst": ["2026-01-27 19:00:00Z"]}).to_csv(
        path, index=False
    )
    return path


def test_unreachable_keeps_the_cached_schedule_when_asked(unreachable, cached, caplog):
    before = cached.read_bytes()

    with caplog.at_level("WARNING"):
        schedule = fetch_league_schedule(
            season="2025-26", output_path=cached, keep_cached_on_network_error=True
        )

    assert cached.read_bytes() == before
    assert schedule["gameId"].tolist() == [22500664]
    assert "keeping the cached" in caplog.text


def test_unreachable_still_fails_by_default(unreachable, cached):
    """A standalone pull must say it failed — only daily-cycle opts into the fallback."""
    with pytest.raises(requests.RequestException):
        fetch_league_schedule(season="2025-26", output_path=cached)


def test_unreachable_with_nothing_cached_fails(unreachable, tmp_path):
    """No file to fall back to means no schedule at all: that cannot be papered over."""
    with pytest.raises(requests.RequestException):
        fetch_league_schedule(
            season="2025-26",
            output_path=tmp_path / "missing.csv",
            keep_cached_on_network_error=True,
        )
