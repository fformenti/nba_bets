"""Put specific past games back in the pending queue, described from the release.

The selector only finds games on the current season's schedule. A game from an
earlier season that history lacks, or holds wrongly as postponed, can never be
selected again: 2024/25 lost five makeup games and kept two played games as
0-0 postponed rows, because ``Games.csv`` had them that way and nothing
upstream can rebuild a past schedule.

This writes a pending payload for each requested gameId, dated from the
release's schedule (the day the game was played) and enriched exactly as the
selector enriches its own (``upcoming_games.build_pending_payloads``). The
normal path then takes over: ``fetch-game-results`` asks a results source,
``append-game-results`` writes history. History is still written by
``append-game-results`` alone.

Run ``fetch-game-results`` straight after: ``fetch-upcoming-games`` prunes any
pending payload outside its own next slate.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.paths import INGESTED_GAMES_UPDATED_HISTORY_PATH, UPCOMING_GAMES_DIR
from src.etl.collectors.boxscores import SCHEDULE
from src.etl.collectors.sportsdataverse import download_table, make_session, season_end_year
from src.etl.collectors.upcoming_games import _write_payload_if_changed, build_pending_payloads
from src.etl.utils.common import get_nba_season
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _team_names(history: pd.DataFrame) -> pd.DataFrame:
    """(teamId, season) → city and name, as history spells them."""
    sides = [
        history[["season", f"{side}teamId", f"{side}teamPrename", f"{side}teamName"]].set_axis(
            ["season", "teamId", "city", "name"], axis=1
        )
        for side in ("home", "away")
    ]
    return (
        pd.concat(sides)
        .dropna(subset=["teamId"])
        .drop_duplicates(["teamId", "season"], keep="last")
    )


def requeue_rows(
    game_ids: list[int], schedule: pd.DataFrame, history: pd.DataFrame
) -> pd.DataFrame:
    """Schedule-shaped rows for ``game_ids``, the selector's input format.

    The release has the date but no tip-off time. A history row on the same
    date supplies its time; otherwise the time is unknown and set to midnight.
    It only orders the game among other games that day, none of which can
    involve either team.
    """
    padded = {f"{game_id:010d}": game_id for game_id in game_ids}
    rows = schedule[schedule["game_id"].isin(padded)]
    missing = sorted(set(padded) - set(rows["game_id"]))
    if missing:
        raise ValueError(f"Not in the release schedule: {missing}")

    known = history.set_index("gameId")["gameDate"]
    names = _team_names(history)
    records = []
    for row in rows.itertuples():
        game_id = padded[row.game_id]
        tipoff = pd.Timestamp(row.game_date)
        if game_id in known.index and pd.Timestamp(known[game_id]).normalize() == tipoff:
            tipoff = pd.Timestamp(known[game_id])
        else:
            logger.warning(f"Game {game_id}: tip-off time unknown, recorded at midnight")
        season = get_nba_season(tipoff)
        record = {
            "gameId": game_id,
            "gameDateTimeEstLocal": tipoff,
            "gameDateOnlyStr": tipoff.strftime("%Y-%m-%d"),
            "hometeamId": int(row.home_team_id),
            "awayteamId": int(row.away_team_id),
            "arenaCity": None,
            "arenaName": None,
            "gameLabel": None,
            "gameSubLabel": None,
        }
        for side, team_id in (("home", row.home_team_id), ("away", row.away_team_id)):
            match = names[(names["teamId"] == team_id) & (names["season"] == season)]
            if match.empty:
                raise ValueError(f"Game {game_id}: team {team_id} has no {season} history row")
            record[f"{side}TeamCity"] = match.iloc[0]["city"]
            record[f"{side}TeamName"] = match.iloc[0]["name"]
        records.append(record)
    return pd.DataFrame(records)


def requeue_games(
    game_ids: list[int],
    output_dir: Path = UPCOMING_GAMES_DIR,
    history_path: Path = INGESTED_GAMES_UPDATED_HISTORY_PATH,
) -> list[Path]:
    """Write a pending payload per game. Refuses a game history already has as played."""
    history = pd.read_csv(history_path, parse_dates=["gameDate"], low_memory=False)
    played = history[pd.to_numeric(history["postponed"], errors="coerce").fillna(0) != 1]
    already = sorted(set(game_ids) & set(played["gameId"]))
    if already:
        raise ValueError(f"Already played in history, not requeued: {already}")

    session = make_session()
    schedules = [
        download_table(session, SCHEDULE, end_year).to_pandas()
        for end_year in sorted({season_end_year(game_id) for game_id in game_ids})
    ]
    rows = requeue_rows(game_ids, pd.concat(schedules, ignore_index=True), history)
    payloads = build_pending_payloads(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for payload in payloads.to_dict("records"):
        path = output_dir / f"{payload['gameId']}.json"
        _write_payload_if_changed(payload, path)
        written.append(path)
        logger.info(f"Requeued {payload['gameId']} for {payload['gameDateOnlyStr']}")
    return written
