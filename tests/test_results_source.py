"""The results-source seam, exercised without a live feed.

Retrieval of finished-game outcomes is behind an interface because the provider
is not settled yet. These tests pin the contract every source must satisfy, and
the one conversion that is easy to get silently wrong: ``winner`` is the winning
*teamId*, not a 0/1 flag.
"""

import json

import pandas as pd
import pytest

from src.config.constants import MAX_FETCH_ATTEMPTS, SCHEDULE_TIMEZONE
from src.etl.collectors.results import SOURCES, get_results_source
from src.etl.collectors.results.base import GameResult, GameStatus, ResultsSource
from src.etl.collectors.results.nba_api_source import _parse_game_status
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


class TestParseGameStatus:
    """Status is read from the provider, never inferred from the scoreline.

    Inferring it is what recorded games that had not tipped off yet as
    permanently postponed: 0-0 is the normal state of every game before it
    starts, so the scoreline cannot tell the two apart.
    """

    @pytest.mark.parametrize("text", ["PPD", "Postponed", "Cancelled", "Suspended"])
    def test_the_text_announces_a_game_that_did_not_happen(self, text):
        assert _parse_game_status(1, text, 0, 0) is GameStatus.POSTPONED

    def test_a_scheduled_game_is_not_postponed(self):
        assert _parse_game_status(1, "7:30 pm ET", 0, 0) is GameStatus.SCHEDULED

    def test_a_game_in_progress_is_not_final(self):
        assert _parse_game_status(2, "Q3 4:12", 54, 61) is GameStatus.IN_PROGRESS

    def test_a_finished_game_is_final(self):
        assert _parse_game_status(3, "Final", 112, 99) is GameStatus.FINAL

    def test_final_without_a_score_is_not_trusted(self):
        """Neither final nor postponed — both would be a guess."""
        assert _parse_game_status(3, "Final", 0, 0) is GameStatus.UNKNOWN

    def test_an_unreadable_status_is_unknown(self):
        assert _parse_game_status(None, None, 0, 0) is GameStatus.UNKNOWN

    def test_only_settled_statuses_are_terminal(self):
        terminal = {status for status in GameStatus if status.is_terminal}
        assert terminal == {GameStatus.FINAL, GameStatus.POSTPONED}


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

    outcomes = enrich_upcoming_games_results(
        upcoming_dir,
        output_dir,
        source=PlaceholderResultsSource(manual_results_dir),
        delay_seconds=0,
    )

    assert len(outcomes["final"]) == 1
    enriched = json.loads(outcomes["final"][0].read_text())
    assert enriched["homeTeamFinalScore"] == 112
    assert enriched["winner"] == HOME_TEAM_ID, "winner must be the winning teamId"
    assert enriched["postponed"] == 0

    # A played game is no longer pending.
    assert not (upcoming_dir / f"{GAME_ID}.json").exists()


def test_unplayed_game_stays_pending(upcoming_dir, manual_results_dir, tmp_path):
    """An unreachable source must leave the payload alone for the next run."""
    outcomes = enrich_upcoming_games_results(
        upcoming_dir,
        tmp_path / "results",
        source=PlaceholderResultsSource(manual_results_dir),
        delay_seconds=0,
    )

    assert outcomes["final"] == []
    assert outcomes["pending"] == [upcoming_dir / f"{GAME_ID}.json"]
    assert (upcoming_dir / f"{GAME_ID}.json").exists()


class TestStatusRouting:
    """Where a payload ends up is decided by status, never by the scoreline."""

    @staticmethod
    def _run(upcoming_dir, manual_results_dir, tmp_path, **kwargs):
        return enrich_upcoming_games_results(
            upcoming_dir,
            tmp_path / "results",
            source=PlaceholderResultsSource(manual_results_dir),
            delay_seconds=0,
            postponed_dir=tmp_path / "postponed",
            unresolved_dir=tmp_path / "unresolved",
            **kwargs,
        )

    def test_postponed_game_is_parked_not_recorded(
        self, upcoming_dir, manual_results_dir, tmp_path
    ):
        """A postponed game must never become a history row."""
        (manual_results_dir / f"{GAME_ID}.json").write_text(
            json.dumps({"status": "postponed"})
        )

        outcomes = self._run(upcoming_dir, manual_results_dir, tmp_path)

        assert outcomes["final"] == []
        assert len(outcomes["postponed"]) == 1
        assert outcomes["postponed"][0].parent.name == "postponed"
        assert not (upcoming_dir / f"{GAME_ID}.json").exists()
        assert not (tmp_path / "results" / f"{GAME_ID}.json").exists()

    def test_a_game_that_has_not_tipped_off_stays_pending(
        self, upcoming_dir, manual_results_dir, tmp_path
    ):
        """0-0 with a scheduled status is not postponed — it is just early."""
        (manual_results_dir / f"{GAME_ID}.json").write_text(
            json.dumps({"status": "scheduled"})
        )

        outcomes = self._run(upcoming_dir, manual_results_dir, tmp_path)

        assert outcomes["postponed"] == []
        assert outcomes["final"] == []
        assert (upcoming_dir / f"{GAME_ID}.json").exists()

    def test_attempts_accumulate_then_quarantine(
        self, upcoming_dir, manual_results_dir, tmp_path
    ):
        """A game the source never settles must stop blocking the pipeline.

        The fixture's tip-off is in the past, so every run here is a genuine
        "the source should know by now and does not" — the only case that is
        allowed to spend an attempt.
        """
        (manual_results_dir / f"{GAME_ID}.json").write_text(
            json.dumps({"status": "scheduled"})
        )
        pending = upcoming_dir / f"{GAME_ID}.json"

        for expected in (1, 2):
            self._run(upcoming_dir, manual_results_dir, tmp_path, max_attempts=3)
            assert json.loads(pending.read_text())["_fetchAttempts"] == expected

        outcomes = self._run(upcoming_dir, manual_results_dir, tmp_path, max_attempts=3)

        assert len(outcomes["quarantined"]) == 1
        assert outcomes["quarantined"][0].parent.name == "unresolved"
        assert not pending.exists()

    def test_reruns_before_tipoff_never_spend_an_attempt(
        self, upcoming_dir, manual_results_dir, tmp_path
    ):
        """Re-running the cycle must not quarantine a slate that has not played.

        The attempt budget used to be spent per invocation, so running
        ``make daily-cycle`` a handful of times on the morning of a slate parked
        every one of that evening's games in ``unresolved/``.
        """
        (manual_results_dir / f"{GAME_ID}.json").write_text(
            json.dumps({"status": "scheduled"})
        )
        pending = upcoming_dir / f"{GAME_ID}.json"
        payload = json.loads(pending.read_text())
        future = pd.Timestamp.now(tz=SCHEDULE_TIMEZONE).tz_localize(None) + pd.Timedelta(
            hours=6
        )
        payload["gameDate"] = future.isoformat()
        pending.write_text(json.dumps(payload))
        before = pending.read_text()

        for _ in range(MAX_FETCH_ATTEMPTS + 1):
            outcomes = self._run(upcoming_dir, manual_results_dir, tmp_path)
            assert outcomes["quarantined"] == []

        assert pending.exists()
        assert "_fetchAttempts" not in json.loads(pending.read_text())
        assert pending.read_text() == before, "an untipped game's payload is untouched"

    def test_unknown_status_is_never_written_as_a_result(
        self, upcoming_dir, manual_results_dir, tmp_path
    ):
        """An ambiguous answer must not be committed as a played game."""
        (manual_results_dir / f"{GAME_ID}.json").write_text(
            json.dumps({"status": "unknown"})
        )

        outcomes = self._run(upcoming_dir, manual_results_dir, tmp_path)

        assert outcomes["final"] == []
        assert outcomes["postponed"] == []
        assert (upcoming_dir / f"{GAME_ID}.json").exists()
