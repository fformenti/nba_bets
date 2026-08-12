"""Re-pull the league schedule from stats.nba.com.

The schedule used to be a file dropped in by hand, which meant it froze the day
it was downloaded. That is fine for games that go ahead as planned and useless
for the ones that do not: a postponed game only gets a makeup date when the
league publishes one, and a frozen file can never show it. Nothing could watch a
postponed game back into the pipeline.

The output keeps the exact column set of the hand-dropped file, so
``parse_league_schedule`` and everything downstream are unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import scheduleleaguev2

from src.config.constants import CURRENT_SEASON_START_YEAR
from src.config.paths import LEAGUE_SCHEDULE_PATH
from src.etl.utils.common import atomic_write_csv
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

NBA_API_TIMEOUT = 90
LEAGUE_ID = "00"

# The hand-dropped file's schema, preserved exactly.
SCHEDULE_COLUMNS = [
    "gameId",
    "gameDateTimeEst",
    "homeTeamId",
    "awayTeamId",
    "homeTeamCity",
    "homeTeamName",
    "awayTeamCity",
    "awayTeamName",
    "gameDay",
    "arenaName",
    "arenaCity",
    "arenaState",
    "gameLabel",
    "gameSubLabel",
    "gameSubtype",
    "seriesGameNumber",
    "weekNumber",
]

# A pull returning far fewer games than the file it replaces means a truncated
# or partial response, not a shorter season. Refuse rather than destroy it.
MIN_RETAINED_FRACTION = 0.9


def season_string(start_year: int = CURRENT_SEASON_START_YEAR) -> str:
    """``2025`` → ``"2025-26"``, the format the endpoint expects."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _flatten_games(league_schedule: dict) -> list[dict]:
    rows = []
    for game_date in league_schedule.get("gameDates", []):
        for game in game_date.get("games", []):
            home = game.get("homeTeam") or {}
            away = game.get("awayTeam") or {}
            rows.append(
                {
                    "gameId": int(game["gameId"]),
                    "gameDateTimeEst": game.get("gameDateTimeEst"),
                    "homeTeamId": home.get("teamId"),
                    "awayTeamId": away.get("teamId"),
                    "homeTeamCity": home.get("teamCity"),
                    "homeTeamName": home.get("teamName"),
                    "awayTeamCity": away.get("teamCity"),
                    "awayTeamName": away.get("teamName"),
                    "gameDay": game.get("day"),
                    "arenaName": game.get("arenaName"),
                    "arenaCity": game.get("arenaCity"),
                    "arenaState": game.get("arenaState"),
                    "gameLabel": game.get("gameLabel"),
                    "gameSubLabel": game.get("gameSubLabel"),
                    "gameSubtype": game.get("gameSubtype"),
                    "seriesGameNumber": game.get("seriesGameNumber"),
                    "weekNumber": game.get("weekNumber"),
                }
            )
    return rows


def fetch_league_schedule(
    season: str | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Download the league schedule and write it to ``output_path``.

    The write is atomic and refuses to shrink the file materially: a bad pull
    must not take the schedule with it, because the schedule is what tells the
    collector which games exist at all.
    """
    season = season or season_string()
    output_path = output_path or LEAGUE_SCHEDULE_PATH

    logger.info(f"Fetching the {season} league schedule")
    response = scheduleleaguev2.ScheduleLeagueV2(
        season=season, league_id=LEAGUE_ID, timeout=NBA_API_TIMEOUT
    )
    league_schedule = response.get_dict().get("leagueSchedule", {})

    rows = _flatten_games(league_schedule)
    if not rows:
        raise RuntimeError(
            f"The {season} schedule came back empty; leaving {output_path.name} alone."
        )

    schedule = pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)
    schedule["gameDateTimeEst"] = pd.to_datetime(
        schedule["gameDateTimeEst"], utc=True, errors="coerce"
    ).dt.strftime("%Y-%m-%d %H:%M:%SZ")
    schedule = schedule.sort_values(["gameDateTimeEst", "gameId"]).reset_index(drop=True)

    _guard_against_shrinkage(schedule, output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(schedule, output_path)
    logger.info(
        "Saved %d scheduled games to %s (%s → %s)",
        len(schedule),
        output_path,
        schedule["gameDateTimeEst"].min()[:10],
        schedule["gameDateTimeEst"].max()[:10],
    )
    return schedule


def _guard_against_shrinkage(schedule: pd.DataFrame, output_path: Path) -> None:
    if not output_path.exists():
        return
    existing = len(pd.read_csv(output_path))
    if len(schedule) < existing * MIN_RETAINED_FRACTION:
        raise RuntimeError(
            f"Refusing to overwrite {output_path.name}: the pull has {len(schedule)} "
            f"games against {existing} on disk. Investigate before replacing it."
        )


