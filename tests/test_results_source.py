"""The results-source seam, exercised without a live feed.

Retrieval of finished-game outcomes is behind an interface because the provider
is not settled yet. These tests pin the contract every source must satisfy, and
the one conversion that is easy to get silently wrong: ``winner`` is the winning
*teamId*, not a 0/1 flag.
"""

import json

import pytest

from src.etl.collectors.results import SOURCES, get_results_source
from src.etl.collectors.results.base import GameResult, ResultsSource
from src.etl.collectors.results.placeholder_source import PlaceholderResultsSource
from src.etl.collectors.upcoming_games_results import (
    enrich_upcoming_games_results,
    resolve_winner_team_id,
)

HOME_TEAM_ID = 1610612741
AWAY_TEAM_ID = 1610612748
GAME_ID = 22500798


@pytest.fixture
def manual_results_dir(tmp_path):
    d = tmp_path / "manual_results"
    d.mkdir()
    return d


@pytest.fixture
def upcoming_dir(tmp_path):
    """One pending upcoming-game payload, as the collector writes them."""
    d = tmp_path / "upcoming_games"
    d.mkdir()
    (d / f"{GAME_ID}.json").write_text(
        json.dumps(
            {
                "gameId": GAME_ID,
                "gameDate": "2026-02-09T00:00:00.000Z",
                "gameDateOnlyStr": "2026-02-09",
                "hometeamId": HOME_TEAM_ID,
                "awayteamId": AWAY_TEAM_ID,
                "season": "2025/26",
            }
        )
    )
    return d


def test_every_registered_source_satisfies_the_protocol():
    for name in SOURCES:
        assert isinstance(get_results_source(name), ResultsSource), name


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="Unknown results source"):
        get_results_source("bookmaker-we-have-not-written-yet")


def test_placeholder_returns_none_for_unplayed_game(manual_results_dir):
    """A missing file means 'not played yet' — it must not raise."""
    source = PlaceholderResultsSource(manual_results_dir)
    assert source.fetch(f"{GAME_ID:010d}") is None


def test_placeholder_reads_a_dropped_result(manual_results_dir):
    (manual_results_dir / f"{GAME_ID}.json").write_text(
        json.dumps({"homeTeamFinalScore": 112, "awayTeamFinalScore": 99})
    )
    result = PlaceholderResultsSource(manual_results_dir).fetch(f"{GAME_ID:010d}")

    assert result.home_score == 112
    assert result.away_score == 99


def test_placeholder_accepts_padded_or_unpadded_ids(manual_results_dir):
    """gameIds appear both ways upstream; both spellings must resolve."""
    (manual_results_dir / f"{GAME_ID:010d}.json").write_text(
        json.dumps({"homeTeamFinalScore": 100, "awayTeamFinalScore": 90})
    )
    source = PlaceholderResultsSource(manual_results_dir)

    assert source.fetch(str(GAME_ID)) is not None
    assert source.fetch(f"{GAME_ID:010d}") is not None


def test_placeholder_rejects_a_malformed_file(manual_results_dir):
    (manual_results_dir / f"{GAME_ID}.json").write_text(json.dumps({"score": "112-99"}))

    with pytest.raises(ValueError, match="homeTeamFinalScore"):
        PlaceholderResultsSource(manual_results_dir).fetch(str(GAME_ID))


class TestResolveWinnerTeamId:
    """`winner` is a teamId. Sources reporting only scores get it filled in."""

    payload = {"hometeamId": HOME_TEAM_ID, "awayteamId": AWAY_TEAM_ID}

    def test_home_win_resolves_to_home_team_id(self):
        result = GameResult(home_score=112, away_score=99)
        assert resolve_winner_team_id(self.payload, result) == HOME_TEAM_ID

    def test_away_win_resolves_to_away_team_id(self):
        result = GameResult(home_score=99, away_score=112)
        assert resolve_winner_team_id(self.payload, result) == AWAY_TEAM_ID

    def test_source_supplied_winner_is_preserved(self):
        """nba_api already reports a teamId; it must not be recomputed."""
        result = GameResult(home_score=99, away_score=112, winner=AWAY_TEAM_ID)
        assert resolve_winner_team_id(self.payload, result) == AWAY_TEAM_ID


def test_full_enrichment_round_trip(upcoming_dir, manual_results_dir, tmp_path):
    """The whole collector step, end to end, on the placeholder source."""
    (manual_results_dir / f"{GAME_ID}.json").write_text(
        json.dumps({"homeTeamFinalScore": 112, "awayTeamFinalScore": 99})
    )
    output_dir = tmp_path / "results"

    written = enrich_upcoming_games_results(
        upcoming_dir,
        output_dir,
        source=PlaceholderResultsSource(manual_results_dir),
        delay_seconds=0,
    )

    assert len(written) == 1
    enriched = json.loads(written[0].read_text())
    assert enriched["homeTeamFinalScore"] == 112
    assert enriched["winner"] == HOME_TEAM_ID, "winner must be the winning teamId"

    # A played game is no longer pending.
    assert not (upcoming_dir / f"{GAME_ID}.json").exists()


def test_unplayed_game_stays_pending(upcoming_dir, manual_results_dir, tmp_path):
    """Without an outcome the payload must be left alone for the next run."""
    written = enrich_upcoming_games_results(
        upcoming_dir,
        tmp_path / "results",
        source=PlaceholderResultsSource(manual_results_dir),
        delay_seconds=0,
    )

    assert written == []
    assert (upcoming_dir / f"{GAME_ID}.json").exists()
