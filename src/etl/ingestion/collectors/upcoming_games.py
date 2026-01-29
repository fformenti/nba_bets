"""Collect upcoming games based on the last played date."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    LEAGUE_SCHEDULE_PATH,
    UPCOMING_GAMES_DIR,
    GAMES_FEATURES_PATH,
    TEAMS_CITIES_CONFERENCE_HISTORY_PROCESSED_PATH,
)
from src.etl.ingestion.teams_history import load_teams_history_table
from src.etl.transformation.add_conference import add_conference
from src.etl.utils.common import get_nba_season, add_neutral_court_game_flag
from src.utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


def _load_games_played(games_path: Path) -> pd.DataFrame:
    games = pd.read_csv(
        games_path,
        parse_dates=["gameDate"],
        low_memory=False,
    )
    if games.empty:
        raise ValueError(f"No games found in {games_path}")
    if games["gameDate"].dtype == object:
        games["gameDate"] = pd.to_datetime(games["gameDate"], errors="coerce")
    if games["gameDate"].isna().all():
        raise ValueError(f"All gameDate values are invalid in {games_path}")
    return games


def _load_schedule(schedule_path: Path) -> pd.DataFrame:
    schedule = pd.read_csv(schedule_path)
    schedule = schedule.rename(
        columns={"homeTeamId": "hometeamId", "awayTeamId": "awayteamId"}
    )
    schedule["gameDateTimeEst"] = pd.to_datetime(
        schedule["gameDateTimeEst"], utc=True, errors="coerce"
    )
    schedule["gameDateTimeEstLocal"] = schedule["gameDateTimeEst"].dt.tz_convert(None)
    schedule["gameDate"] = schedule["gameDateTimeEstLocal"].dt.date
    return schedule


def _get_next_upcoming_games(
    schedule: pd.DataFrame, games_played: pd.DataFrame
) -> pd.DataFrame:
    last_played_timestamp = games_played["gameDate"].max()
    last_played_date = last_played_timestamp.date()
    last_played_date_str = last_played_date.strftime("%Y-%m-%d")

    games_played_at_date = games_played[
        games_played["gameDateOnlyStr"] == last_played_date_str
    ].copy()

    schedule["gameDateOnlyStr"] = schedule["gameDate"].apply(
        lambda x: x.strftime("%Y-%m-%d")
    )
    schedule_at_date = schedule[
        schedule["gameDateOnlyStr"] == last_played_date_str
    ].copy()

    remaining_schedule_at_date = schedule_at_date[
        ~schedule_at_date["gameId"].isin(games_played_at_date["gameId"])
    ]
    print("count of remaining_schedule_at_date", len(remaining_schedule_at_date))
    if remaining_schedule_at_date.empty:
        remaining_schedule = schedule[schedule["gameDate"] > last_played_date]
        if remaining_schedule.empty:
            logger.warning("No upcoming games found after last played date.")
            return remaining_schedule
        next_date = remaining_schedule["gameDate"].min()
        upcoming_games = remaining_schedule[remaining_schedule["gameDate"] == next_date]
        return upcoming_games
    return remaining_schedule_at_date


def _save_upcoming_games(upcoming_games: pd.DataFrame, output_path: Path) -> None:
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
    return


def collect_upcoming_games() -> None:
    schedule = _load_schedule(LEAGUE_SCHEDULE_PATH)
    games_played = _load_games_played(GAMES_FEATURES_PATH)
    # To do - Create a script called process schedule and rename raw_games.py to process_raw_games.py
    schedule["gameDateOnlyStr"] = schedule["gameDate"].apply(
        lambda x: x.strftime("%Y-%m-%d")
    )
    upcoming_games = _get_next_upcoming_games(schedule, games_played)

    upcoming_games["season"] = upcoming_games["gameDateTimeEstLocal"].apply(
        get_nba_season
    )
    teams_history = load_teams_history_table(
        TEAMS_CITIES_CONFERENCE_HISTORY_PROCESSED_PATH
    )
    upcoming_games = add_conference(
        upcoming_games, teams_history, ignore_winner_column=True
    )

    # Add neutral court game flag (handles missing gameLabel column gracefully)
    upcoming_games = add_neutral_court_game_flag(
        upcoming_games, game_label_column="gameLabel", drop_label_column=True
    )

    upcoming_games = upcoming_games.drop(columns=["gameDate"], errors="ignore")
    upcoming_games = upcoming_games.rename(columns={"gameDateTimeEstLocal": "gameDate"})
    upcoming_games = upcoming_games[
        [
            "gameId",
            "gameDate",
            "gameDateOnlyStr",
            "arenaCity",
            "arenaName",
            "hometeamId",
            "homeTeamCity",
            "homeTeamName",
            "hometeamConference",
            "awayteamId",
            "awayTeamCity",
            "awayTeamName",
            "awayteamConference",
            "winnerteamConference",
            "season",
            "is_neutral_court_game",
        ]
    ].copy()

    _save_upcoming_games(upcoming_games, UPCOMING_GAMES_DIR)
    logger.info(
        "upcoming games: %s",
        upcoming_games,
    )
    return


def main() -> None:
    setup_logging(level="INFO")

    collect_upcoming_games()


if __name__ == "__main__":
    main()
