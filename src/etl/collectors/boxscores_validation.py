"""Check stored box scores against themselves and against our game history.

Report-only: every check returns a count, and nothing here raises on a bad
number. Schema drift is the one fatal check, and it runs at download time in
``sportsdataverse.check_schema``. The counts are written per season to
``BOXSCORES_VALIDATION_PATH``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.paths import (
    INGESTED_GAMES_UPDATED_HISTORY_PATH,
    RAW_BOXSCORES_MANIFEST_PATH,
    RAW_BOXSCORES_UNPLAYED_PATH,
)
from src.etl.collectors.boxscores import PLAYER, TEAM
from src.etl.collectors.sportsdataverse import ReleaseFile, season_label
from src.etl.utils.common import game_type_from_id, read_json
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

REGULAR, PLAYOFFS, PLAY_IN, CUP_FINAL = "2", "4", "5", "6"

# Regular-season games a full season has: 29 teams x 82 / 2 until the Bobcats
# joined in 2004/05, then 30 x 82 / 2. Shortened seasons as played.
SHORT_SEASON_GAMES = {
    1999: 725,  # 1998/99 lockout, 50 games a team
    2012: 990,  # 2011/12 lockout, 66 games a team
    2020: 1059,  # 2019/20 COVID: 971 before the pause + 88 bubble seeding games
    2021: 1080,  # 2020/21, 72 games a team
}

_FETCH_COUNTS = ["dropped_undated_games", "dropped_unplayed_games", "dropped_future_games"]


def expected_regular_games(end_year: int) -> int:
    if end_year in SHORT_SEASON_GAMES:
        return SHORT_SEASON_GAMES[end_year]
    return 1189 if end_year <= 2004 else 1230


def load_history_games(path: Path = INGESTED_GAMES_UPDATED_HISTORY_PATH) -> pd.DataFrame:
    """Non-preseason history games: ``game_id`` (10-char str), ``season``, ``game_date``."""
    history = pd.read_csv(path, usecols=["gameId", "season", "gameDate"])
    history = history[history["gameId"].map(game_type_from_id) != "Preseason"]
    return pd.DataFrame(
        {
            "game_id": history["gameId"].map(lambda game_id: f"{int(game_id):010d}"),
            "season": history["season"],
            "game_date": pd.to_datetime(history["gameDate"]).dt.normalize(),
        }
    )


def load_season(kind: ReleaseFile, end_year: int) -> pd.DataFrame | None:
    path = kind.path(end_year)
    return pd.read_parquet(path) if path.exists() else None


def validate_season(
    end_year: int,
    team: pd.DataFrame,
    player: pd.DataFrame,
    history: pd.DataFrame,
    all_box_game_ids: frozenset[str] = frozenset(),
) -> dict:
    """Counts for one season.

    ``history`` holds that season's ``game_id`` and ``game_date``. ``all_box_game_ids``,
    every season's, tells a game missing from the box scores apart from one the
    two sources file under different seasons: history labels a season by date,
    so the 2020 bubble playoffs (September-October) land in 2020/21.
    """
    team_games = set(team["game_id"])
    player_games = set(player["game_id"])
    games_by_type = team.groupby("season_type_id")["game_id"].nunique()

    rows_per_game = team.groupby("game_id").size()
    sides_per_game = team.groupby("game_id")["side"].agg(frozenset)

    both = team_games & player_games
    team_points = team[team["game_id"].isin(both)].set_index(["game_id", "team_id"])["points"]
    player_points = (
        player[player["game_id"].isin(both)].groupby(["game_id", "team_id"])["points"].sum()
    )
    points = pd.concat([team_points, player_points], axis=1, keys=["team", "player"])
    mismatched = points[points["team"] != points["player"]].index.get_level_values("game_id")

    # The ordering check: does history date each game on the day it was played?
    box_dates = team.drop_duplicates("game_id").set_index("game_id")["game_date"]
    history_dates = history.drop_duplicates("game_id").set_index("game_id")["game_date"]
    shared = box_dates.index.intersection(history_dates.index)
    misdated = shared[box_dates[shared] != history_dates[shared]]

    history_games = set(history["game_id"])
    missing_from_box = sorted(history_games - team_games)
    for name, offenders in [
        ("history games not in the team box scores", missing_from_box),
        ("games not exactly two team rows", rows_per_game[rows_per_game != 2].index),
        ("games with team points != player points", mismatched.unique()),
        ("games history dates differently from the played date", misdated),
    ]:
        if len(offenders):
            logger.warning(
                f"{season_label(end_year)}: {len(offenders)} {name}, e.g. {sorted(offenders)[:10]}"
            )

    return {
        "season": season_label(end_year),
        "end_year": end_year,
        "team_rows": len(team),
        "player_rows": len(player),
        "games": len(team_games),
        "regular_games": int(games_by_type.get(REGULAR, 0)),
        "expected_regular_games": expected_regular_games(end_year),
        "playoff_games": int(games_by_type.get(PLAYOFFS, 0)),
        "play_in_games": int(games_by_type.get(PLAY_IN, 0)),
        "cup_final_games": int(games_by_type.get(CUP_FINAL, 0)),
        "first_game_date": box_dates.min().date() if len(box_dates) else None,
        "last_game_date": box_dates.max().date() if len(box_dates) else None,
        "games_not_two_team_rows": int((rows_per_game != 2).sum()),
        "games_without_home_and_away": int((sides_per_game != frozenset({"home", "away"})).sum()),
        "games_team_only": len(team_games - player_games),
        "games_player_only": len(player_games - team_games),
        "points_mismatch_games": int(mismatched.nunique()),
        "history_games": len(history_games),
        "history_found": len(history_games & team_games),
        "history_coverage": (
            round(len(history_games & team_games) / len(history_games), 4)
            if history_games
            else None
        ),
        "history_in_other_box_season": len(set(missing_from_box) & all_box_game_ids),
        "box_games_not_in_history": len(team_games - history_games),
        "history_date_mismatches": len(misdated),
    }


def validate_boxscores(
    seasons: list[int],
    history: pd.DataFrame,
    team_kind: ReleaseFile = TEAM,
    player_kind: ReleaseFile = PLAYER,
    manifest_path: Path = RAW_BOXSCORES_MANIFEST_PATH,
    unplayed_path: Path = RAW_BOXSCORES_UNPLAYED_PATH,
) -> pd.DataFrame:
    """One row per stored season. A season missing either file is skipped and logged.

    What the fetch dropped (undated, unplayed and future games) comes from the
    manifest, and cancelled or postponed games from the unplayed record, because
    none of those games are in the stored files.
    """
    stored = {}
    for end_year in seasons:
        team = load_season(team_kind, end_year)
        player = load_season(player_kind, end_year)
        if team is None or player is None:
            logger.warning(f"{season_label(end_year)}: box scores not stored, not validated")
            continue
        stored[end_year] = (team, player)

    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    unplayed = (
        pd.read_csv(unplayed_path, dtype={"game_id": str})
        if unplayed_path.exists()
        else pd.DataFrame(columns=["end_year", "status"])
    )

    all_box_game_ids = frozenset().union(*(set(team["game_id"]) for team, _ in stored.values()))
    rows = []
    for end_year, (team, player) in stored.items():
        season_history = history[history["season"] == season_label(end_year)]
        row = validate_season(end_year, team, player, season_history, all_box_game_ids)
        entry = manifest.get(team_kind.name, {}).get(str(end_year), {})
        row.update({count: entry.get(count) for count in _FETCH_COUNTS})
        season_unplayed = unplayed[unplayed["end_year"] == end_year]
        row["cancelled_games"] = int((season_unplayed["status"] == "cancelled").sum())
        row["postponed_games"] = int((season_unplayed["status"] == "postponed").sum())
        rows.append(row)
    return pd.DataFrame(rows)
