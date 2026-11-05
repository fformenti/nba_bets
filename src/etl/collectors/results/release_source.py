"""Results from the sportsdataverse releases: final scores and the played date.

Reads two of the releases ``boxscores.py`` documents, for the season a gameId
belongs to:

    nba_stats_schedules        nba_schedule_{Y}.parquet     date, scores, W/L
    nba_stats_team_boxscores   team_boxscores_{Y}.parquet   minutes, for overtimes

Both are downloaded once per season per run and kept in memory. Unlike
stats.nba.com, which times out from the laptop, this needs only GitHub.

What it can say about a game, from the schedule row:

- W/L on both sides → ``FINAL``, with ``game_date`` the day it was played. A
  makeup game carries its makeup date, which is the point of this source.
- No result, dated after today → ``SCHEDULED``, with its current date. The
  router parks the game as postponed if that date is later than the payload's
  and the original tip-off has passed.
- No result, dated in the past, in a season that is over → ``CANCELLED``.
- No result, dated in the past, this season → ``UNKNOWN``: postponed with no
  new date yet, or simply not rebuilt yet. Attempts are spent, and the game is
  quarantined if it stays that way.

The release has no attendance and no inactive list, so both stay empty. The
release lags the games by up to a day (rebuilt about daily), which only delays
a result; it never invents one.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import requests

from src.etl.collectors.boxscores import SCHEDULE, TEAM
from src.etl.collectors.results.base import GameResult, GameStatus
from src.etl.collectors.sportsdataverse import (
    ReleaseFile,
    SchemaDriftError,
    current_season_end_year,
    download_table,
    make_session,
    season_end_year,
    today_eastern,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

REGULATION_MINUTES = 240  # 5 players x 48 minutes
OVERTIME_MINUTES = 25  # 5 players x 5 minutes


def overtimes_from_minutes(minutes: str | None) -> int | None:
    """``"265:00"`` → 1. A team's minutes are 240 plus 25 per overtime period."""
    if not minutes:
        return None
    try:
        played = int(str(minutes).split(":")[0])
    except ValueError:
        return None
    return max(0, round((played - REGULATION_MINUTES) / OVERTIME_MINUTES))


class ReleaseResultsSource:
    """Fetch outcomes from the sportsdataverse GitHub releases."""

    name = "release"

    def __init__(
        self,
        session: requests.Session | None = None,
        today: date | None = None,
        schedule: ReleaseFile = SCHEDULE,
        team: ReleaseFile = TEAM,
    ):
        self._session = session or make_session()
        self._today = today or today_eastern()
        self._schedule = schedule
        self._team = team
        self._seasons: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}

    def _season(self, end_year: int) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        if end_year not in self._seasons:
            try:
                schedule = download_table(self._session, self._schedule, end_year).to_pandas()
                team = download_table(self._session, self._team, end_year).to_pandas()
            except (requests.RequestException, SchemaDriftError) as error:
                logger.warning(f"Release data for season {end_year} unavailable: {error}")
                return None
            self._seasons[end_year] = (
                schedule.set_index("game_id"),
                team.groupby("game_id")["minutes"].first(),
            )
        return self._seasons[end_year]

    def fetch(self, game_id: str) -> GameResult | None:
        padded = f"{int(game_id):010d}"
        end_year = season_end_year(padded)
        season = self._season(end_year)
        if season is None:
            return None
        schedule, minutes = season

        if padded not in schedule.index:
            logger.info(f"Game {padded} is not in the {end_year} release")
            return GameResult(status=GameStatus.UNKNOWN)
        row = schedule.loc[padded]
        game_date = str(row["game_date"])[:10]

        if pd.notna(row["home_wl"]) and pd.notna(row["away_wl"]):
            overtimes = overtimes_from_minutes(minutes.get(padded))
            if overtimes is None:
                # The team file is rebuilt minutes after the schedule. Ask again
                # rather than record a guessed overtime count.
                logger.info(f"Game {padded} has a score but no team box yet")
                return GameResult(status=GameStatus.UNKNOWN, game_date=game_date)
            home_won = row["home_wl"] == "W"
            return GameResult(
                home_score=int(row["home_pts"]),
                away_score=int(row["away_pts"]),
                overtimes=overtimes,
                winner=int(row["home_team_id"] if home_won else row["away_team_id"]),
                status=GameStatus.FINAL,
                game_date=game_date,
            )

        return GameResult(status=self._unplayed_status(game_date, end_year), game_date=game_date)

    def _unplayed_status(self, game_date: str, end_year: int) -> GameStatus:
        """A game with no result: still to come, cancelled, or not knowable yet."""
        if pd.Timestamp(game_date).date() > self._today:
            return GameStatus.SCHEDULED
        if end_year < current_season_end_year(self._today):
            return GameStatus.CANCELLED
        return GameStatus.UNKNOWN

    def available(self) -> bool:
        """GitHub is reachable often enough; a failed download returns ``None``."""
        return True
