"""Collect upcoming games based on the last played date."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import (
    LOCAL_LEAGUE_SCHEDULE_PATH,
    LOCAL_PROCESSED_FOLDER,
    PROJECT_ROOT,
)
from src.etl.utils.common import get_nba_season
from src.utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


@dataclass
class UpcomingGamesResult:
    last_played_date: pd.Timestamp
    last_day_games: pd.DataFrame
    upcoming_games: pd.DataFrame


def _load_last_played_date(games_path: Path) -> pd.Timestamp:
    games = pd.read_csv(games_path, usecols=["gameDate"], parse_dates=["gameDate"])
    if games.empty:
        raise ValueError(f"No games found in {games_path}")
    return games["gameDate"].max()


def get_last_played_games(games_path: Path) -> tuple[pd.Timestamp, pd.DataFrame]:
    games = pd.read_csv(games_path, parse_dates=["gameDate"])
    if games.empty:
        raise ValueError(f"No games found in {games_path}")

    last_played_dt = games["gameDate"].max()
    last_played_date = last_played_dt.date()
    last_day_games = games[games["gameDate"].dt.date == last_played_date].copy()
    return last_played_dt, last_day_games


def _parse_schedule(schedule_path: Path) -> pd.DataFrame:
    schedule = pd.read_csv(schedule_path)
    schedule["gameDateTimeEst"] = pd.to_datetime(
        schedule["gameDateTimeEst"], utc=True, errors="coerce"
    )
    schedule["gameDateTimeEstLocal"] = schedule["gameDateTimeEst"].dt.tz_convert(None)
    schedule["gameDate"] = schedule["gameDateTimeEstLocal"].dt.date
    return schedule


def _load_team_history(teams_history_path: Path) -> pd.DataFrame:
    history = pd.read_csv(teams_history_path)
    history = history.drop_duplicates(subset=["teamId", "season"], keep="last")
    return history[["teamId", "teamCity", "teamName", "season"]].copy()


def _attach_team_names(
    schedule_games: pd.DataFrame, teams_history_path: Path
) -> pd.DataFrame:
    schedule_games = schedule_games.copy()
    schedule_games["season"] = schedule_games["gameDateTimeEstLocal"].apply(
        get_nba_season
    )

    team_history = _load_team_history(teams_history_path)

    home_history = team_history.rename(
        columns={
            "teamId": "hometeamId",
            "teamCity": "homeTeamCity",
            "teamName": "homeTeamName",
        }
    )
    away_history = team_history.rename(
        columns={
            "teamId": "awayteamId",
            "teamCity": "awayTeamCity",
            "teamName": "awayTeamName",
        }
    )

    enriched = schedule_games.merge(
        home_history, how="left", on=["hometeamId", "season"]
    ).merge(away_history, how="left", on=["awayteamId", "season"])
    return enriched


def collect_upcoming_games(
    games_path: Path,
    schedule_path: Path = LOCAL_LEAGUE_SCHEDULE_PATH,
    output_path: Path | None = None,
) -> UpcomingGamesResult:
    """
    Find upcoming games on the nearest date to the last played game.
    """
    last_played_dt, last_day_games = get_last_played_games(games_path)
    schedule = _parse_schedule(schedule_path)

    candidate_mask = schedule["gameDateTimeEstLocal"] >= last_played_dt
    candidates = schedule[candidate_mask].copy()
    if candidates.empty:
        candidates = schedule[schedule["gameDate"] >= last_played_dt.date()].copy()

    if candidates.empty:
        raise ValueError("No upcoming games found in the schedule.")

    last_played_date = last_played_dt.date()
    last_day_schedule = schedule[schedule["gameDate"] == last_played_date].copy()
    played_game_ids = set(
        pd.to_numeric(last_day_games["gameId"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )
    schedule_ids = pd.to_numeric(last_day_schedule["gameId"], errors="coerce")
    remaining_last_day = last_day_schedule[~schedule_ids.isin(played_game_ids)].copy()

    if not remaining_last_day.empty:
        upcoming_games = remaining_last_day
    else:
        next_day_candidates = candidates[
            candidates["gameDate"] > last_played_date
        ].copy()
        if next_day_candidates.empty:
            raise ValueError("No upcoming games found after last played date.")
        nearest_date = next_day_candidates["gameDate"].min()
        upcoming_games = next_day_candidates[
            next_day_candidates["gameDate"] == nearest_date
        ].copy()

    teams_history_path = LOCAL_PROCESSED_FOLDER / "teams_history_expanded.csv"
    upcoming_games = _attach_team_names(upcoming_games, teams_history_path)
    upcoming_games = upcoming_games.drop(columns=["gameDate"], errors="ignore")
    upcoming_games = upcoming_games.rename(columns={"gameDateTimeEstLocal": "gameDate"})
    upcoming_games = upcoming_games[
        [
            "gameId",
            "gameDate",
            "arenaCity",
            "arenaName",
            "hometeamId",
            "homeTeamCity",
            "homeTeamName",
            "awayteamId",
            "awayTeamCity",
            "awayTeamName",
        ]
    ].copy()

    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
        for _, row in upcoming_games.iterrows():
            game_id = row["gameId"]
            if pd.isna(game_id):
                continue
            game_id_int = int(game_id)
            game_payload = row.to_dict()
            json_path = output_path / f"{game_id_int}.json"
            pd.Series(game_payload).to_json(json_path, date_format="iso")
        logger.info("Saved upcoming games JSON to %s", output_path)

    return UpcomingGamesResult(
        last_played_date=last_played_dt,
        last_day_games=last_day_games,
        upcoming_games=upcoming_games,
    )


def main() -> None:
    setup_logging(level="INFO")

    games_path = LOCAL_PROCESSED_FOLDER / "regular_season" / "games.csv"
    result = collect_upcoming_games(
        games_path=games_path,
        schedule_path=LOCAL_LEAGUE_SCHEDULE_PATH,
        output_path=PROJECT_ROOT / "data" / "raw" / "incremental" / "upcoming_games",
    )

    logger.info(
        "Last played date: %s | last-day games: %s | upcoming games: %s",
        result.last_played_date.date(),
        len(result.last_day_games),
        len(result.upcoming_games),
    )


if __name__ == "__main__":
    main()
