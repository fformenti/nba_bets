"""Download team and player box scores, one parquet file per season.

Source: the sportsdataverse data releases on GitHub, which republish
stats.nba.com box scores and need no key and no scraping:

    https://github.com/sportsdataverse/sportsdataverse-data/releases/download/<tag>/<file>

    tag nba_stats_team_boxscores     file team_boxscores_{Y}.parquet
    tag nba_stats_player_boxscores   file player_boxscores_{Y}.parquet
    tag nba_stats_schedules          file nba_schedule_{Y}.parquet   (read, not stored)

``{Y}`` is the season's END year: 1997 is our "1996/97". The box files hold
all season types (``season_type_id`` 2 regular, 4 playoffs, 5 play-in, 6 NBA
Cup final) and no date. The schedule file supplies ``game_date``, the date the
game was actually played: a postponed game carries its makeup date there. Join
to our history on ``game_id``, our ``gameId`` zero-padded to 10 characters.

Only played games are stored. The release also lists games that have not been
played: every future game of the current season in the schedule, and the odd
placeholder in the box files (the 2026/27 Cup final, with null stats, a month
before it is played). A game with no result whose date has passed is recorded
in ``RAW_BOXSCORES_UNPLAYED_PATH`` instead: ``cancelled`` once its season is
over, ``postponed`` while the season is current and a makeup date may come.

Each release has one ``timestamp.txt`` for the whole tag, not one per season.
The current season is rebuilt about daily, past seasons rarely (all of them on
2026-09-30), and the team and player files are rebuilt minutes apart, so the
current season's two files can disagree by a game. The schedule release's own
timestamp is not maintained (May 2026 while its 2027 file is rebuilt daily), so
the schedule is simply re-read with every box file it dates.

Terms: the data is NBA statistics, so NBA.com's terms apply through the
republisher (no use "in connection with any gambling activity"; private,
non-commercial use with attribution). GitHub reports MIT for the release repo
while its README says CC BY 4.0. stats.nba.com and cdn.nba.com are not called
directly: they time out or return 403 from the laptop. ESPN and
Basketball-Reference are not used: their terms forbid AI training, and this
project fine-tunes an LLM. Research and checks:
docs/findings/pipeline/2026-11-02-nba-data-sources.md.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import requests

from src.config.paths import (
    RAW_BOXSCORES_MANIFEST_PATH,
    RAW_BOXSCORES_UNPLAYED_PATH,
    RAW_PLAYER_BOXSCORES_DIR,
    RAW_TEAM_BOXSCORES_DIR,
)
from src.etl.collectors.sportsdataverse import (
    ReleaseFile,
    SchemaDriftError,
    all_seasons,
    current_season_end_year,
    download_table,
    make_session,
    release_timestamp,
    season_label,
    today_eastern,
)
from src.etl.utils.common import atomic_write_csv, atomic_write_parquet, read_json
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    "KINDS",
    "PLAYER",
    "SCHEDULE",
    "TEAM",
    "FetchOutcome",
    "SchemaDriftError",
    "fetch_boxscores",
    "needs_download",
    "prepare_season",
    "season_label",
    "unplayed_games",
]

_STAT_COLUMNS = {
    "minutes": "string",
    "field_goals_made": "int64",
    "field_goals_attempted": "int64",
    "field_goals_percentage": "double",
    "three_pointers_made": "int64",
    "three_pointers_attempted": "int64",
    "three_pointers_percentage": "double",
    "free_throws_made": "int64",
    "free_throws_attempted": "int64",
    "free_throws_percentage": "double",
    "rebounds_offensive": "int64",
    "rebounds_defensive": "int64",
    "rebounds_total": "int64",
    "assists": "int64",
    "steals": "int64",
    "blocks": "int64",
    "turnovers": "int64",
    "fouls_personal": "int64",
    "points": "int64",
    "plus_minus_points": "double",
    "game_id": "string",
    "season": "int32",
    "season_type_id": "string",
}
_TEAM_COLUMNS = {
    "team_id": "int64",
    "team_city": "string",
    "team_name": "string",
    "team_tricode": "string",
    "team_slug": "string",
    "side": "string",
}

# The published schemas, in order, as read for every season 1997-2027 on
# 2026-11-04/05. Arrow types, with every string flavour as "string".
TEAM_SCHEMA = {**_TEAM_COLUMNS, **_STAT_COLUMNS}
PLAYER_SCHEMA = {
    **_TEAM_COLUMNS,
    "person_id": "int64",
    "first_name": "string",
    "family_name": "string",
    "name_i": "string",
    "player_slug": "string",
    "position": "string",
    "comment": "string",
    "jersey_num": "string",
    **_STAT_COLUMNS,
}
SCHEDULE_SCHEMA = {
    "game_id": "string",
    "season": "int64",
    "season_type": "string",
    "game_date": "string",
    "matchup": "string",
    "home_team_id": "int64",
    "home_team_abbreviation": "string",
    "home_team_name": "string",
    "home_pts": "int64",
    "home_wl": "string",
    "away_team_id": "int64",
    "away_team_abbreviation": "string",
    "away_team_name": "string",
    "away_pts": "int64",
    "away_wl": "string",
}

TEAM = ReleaseFile(
    "team",
    "nba_stats_team_boxscores",
    "team_boxscores_{year}.parquet",
    TEAM_SCHEMA,
    RAW_TEAM_BOXSCORES_DIR,
)
PLAYER = ReleaseFile(
    "player",
    "nba_stats_player_boxscores",
    "player_boxscores_{year}.parquet",
    PLAYER_SCHEMA,
    RAW_PLAYER_BOXSCORES_DIR,
)
SCHEDULE = ReleaseFile(
    "schedule", "nba_stats_schedules", "nba_schedule_{year}.parquet", SCHEDULE_SCHEMA
)
KINDS = (TEAM, PLAYER)

# Game types whose box scores the releases carry: regular season, playoffs,
# play-in, NBA Cup final. Preseason and All-Star games are not box-scored.
BOX_SCORED_PREFIXES = ("002", "004", "005", "006")

UNPLAYED_COLUMNS = [
    "game_id",
    "season",
    "end_year",
    "listed_date",
    "home_team_id",
    "away_team_id",
    "status",
    "recorded_at",
]


@dataclass(frozen=True)
class FetchOutcome:
    kind: str
    end_year: int
    action: str  # downloaded | skipped | not_published


def needs_download(
    *,
    stored: bool,
    end_year: int,
    current_end_year: int,
    stored_timestamp: str | None,
    release_timestamp: str,
    refresh_all: bool,
) -> bool:
    """Whether a season must be (re)downloaded.

    A missing file is always fetched (the backfill). Past seasons are then left
    alone unless ``refresh_all``, because the timestamp belongs to the whole
    release and moves every day the current season is rebuilt.
    """
    if not stored:
        return True
    if end_year != current_end_year and not refresh_all:
        return False
    return stored_timestamp != release_timestamp


def _schedule_dates(schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per game: its played (or listed) date and whether it has a result."""
    return pd.DataFrame(
        {
            "game_id": schedule["game_id"],
            "game_date": pd.to_datetime(schedule["game_date"]),
            "has_result": schedule["home_wl"].notna() & schedule["away_wl"].notna(),
        }
    )


def prepare_season(
    frame: pd.DataFrame, schedule: pd.DataFrame, end_year: int, today: date
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep the played games of one box file, dated, and count what was dropped.

    A game is dropped when the schedule cannot date it, when either source
    says it has no result (null points, or no W/L), or when it is dated after
    ``today``. Nothing from the future is ever stored.
    """
    dated = frame.merge(_schedule_dates(schedule), on="game_id", how="left")
    undated = dated["game_date"].isna()
    unplayed = ~undated & (~dated["has_result"].astype(bool) | dated["points"].isna())
    future = ~undated & ~unplayed & (dated["game_date"].dt.date > today)

    counts = {
        "dropped_undated_games": int(dated.loc[undated, "game_id"].nunique()),
        "dropped_unplayed_games": int(dated.loc[unplayed, "game_id"].nunique()),
        "dropped_future_games": int(dated.loc[future, "game_id"].nunique()),
    }
    for reason, mask in [
        ("no schedule date", undated),
        ("no result", unplayed),
        ("future date", future),
    ]:
        if mask.any():
            logger.warning(
                f"{season_label(end_year)}: dropping {dated.loc[mask, 'game_id'].nunique()} "
                f"games with {reason}, e.g. {sorted(dated.loc[mask, 'game_id'].unique())[:5]}"
            )

    kept = dated[~(undated | unplayed | future)].drop(columns=["has_result"])
    kept["season_label"] = season_label(end_year)
    return kept.reset_index(drop=True), counts


def unplayed_games(
    schedule: pd.DataFrame, end_year: int, current_end_year: int, today: date
) -> pd.DataFrame:
    """Box-scored games the release has moved past without a result.

    A game whose season is over was cancelled. In the current season a game is
    postponed only when the release already holds results dated after it: the
    release is rebuilt about daily, so last night's games have no result yet
    and are not postponed, just not in yet. Rows without both teams are release
    placeholders, not games.
    """
    dates = _schedule_dates(schedule)
    listed = schedule.assign(game_date=dates["game_date"], has_result=dates["has_result"])
    if end_year < current_end_year:
        cutoff = pd.Timestamp(today)
    else:
        cutoff = listed.loc[listed["has_result"], "game_date"].max()
    missing = listed[
        ~listed["has_result"]
        & (listed["game_date"] < cutoff)
        & listed["game_id"].str[:3].isin(BOX_SCORED_PREFIXES)
        & (listed["home_team_id"] > 0)
        & (listed["away_team_id"] > 0)
    ]
    return pd.DataFrame(
        {
            "game_id": missing["game_id"],
            "season": season_label(end_year),
            "end_year": end_year,
            "listed_date": missing["game_date"].dt.strftime("%Y-%m-%d"),
            "home_team_id": missing["home_team_id"],
            "away_team_id": missing["away_team_id"],
            "status": "cancelled" if end_year < current_end_year else "postponed",
        }
    ).reset_index(drop=True)


def fetch_boxscores(
    seasons: list[int] | None = None,
    *,
    refresh_all: bool = False,
    kinds: tuple[ReleaseFile, ...] = KINDS,
    schedule: ReleaseFile = SCHEDULE,
    manifest_path: Path = RAW_BOXSCORES_MANIFEST_PATH,
    unplayed_path: Path = RAW_BOXSCORES_UNPLAYED_PATH,
    session: requests.Session | None = None,
    today: date | None = None,
) -> list[FetchOutcome]:
    """Backfill missing seasons and refresh the current one when its release moved.

    Each season is written atomically and recorded in the manifest straight
    away, so an interrupted backfill resumes where it stopped.
    """
    session = session or make_session()
    today = today or today_eastern()
    current = current_season_end_year(today)
    seasons = seasons or all_seasons(today)
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    outcomes = []

    timestamps = {}
    for kind in kinds:
        timestamps[kind.name] = release_timestamp(session, kind)
        logger.info(f"{kind.tag}: release timestamp {timestamps[kind.name]}")

    for end_year in seasons:
        due = [
            kind
            for kind in kinds
            if needs_download(
                stored=kind.path(end_year).exists(),
                end_year=end_year,
                current_end_year=current,
                stored_timestamp=manifest.get(kind.name, {})
                .get(str(end_year), {})
                .get("release_timestamp"),
                release_timestamp=timestamps[kind.name],
                refresh_all=refresh_all,
            )
        ]
        outcomes += [
            FetchOutcome(kind.name, end_year, "skipped") for kind in kinds if kind not in due
        ]
        if not due:
            continue

        try:
            season_schedule = download_table(session, schedule, end_year).to_pandas()
        except requests.HTTPError as error:
            if _not_yet_published(error, end_year, current):
                outcomes += [FetchOutcome(kind.name, end_year, "not_published") for kind in due]
                continue
            raise
        _record_unplayed(
            unplayed_games(season_schedule, end_year, current, today), end_year, unplayed_path
        )

        for kind in due:
            try:
                table = download_table(session, kind, end_year)
            except requests.HTTPError as error:
                if _not_yet_published(error, end_year, current):
                    outcomes.append(FetchOutcome(kind.name, end_year, "not_published"))
                    continue
                raise

            frame, counts = prepare_season(table.to_pandas(), season_schedule, end_year, today)
            atomic_write_parquet(frame, kind.path(end_year))
            manifest.setdefault(kind.name, {})[str(end_year)] = {
                "release_timestamp": timestamps[kind.name],
                "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "rows": len(frame),
                "games": int(frame["game_id"].nunique()),
                **counts,
            }
            _write_manifest(manifest, manifest_path)
            logger.info(
                f"{kind.filename(end_year)}: {len(frame)} rows, {frame['game_id'].nunique()} games"
            )
            outcomes.append(FetchOutcome(kind.name, end_year, "downloaded"))

    return outcomes


def _not_yet_published(error: requests.HTTPError, end_year: int, current: int) -> bool:
    """The next season's files appear only once its first game is played."""
    if end_year == current and error.response is not None and error.response.status_code == 404:
        logger.warning(f"{season_label(end_year)} is not published yet")
        return True
    return False


def _record_unplayed(season_unplayed: pd.DataFrame, end_year: int, path: Path) -> None:
    """Replace one season's rows in the unplayed record, keeping first-seen times.

    A game that has since been played drops out because the new rows no longer
    hold it; one that is still unplayed keeps the time it was first recorded.
    """
    existing = (
        pd.read_csv(path, dtype={"game_id": str})
        if path.exists()
        else pd.DataFrame(columns=UNPLAYED_COLUMNS)
    )
    first_seen = dict(zip(existing["game_id"], existing["recorded_at"], strict=True))
    now = datetime.now(UTC).isoformat(timespec="seconds")
    season_unplayed = season_unplayed.assign(
        recorded_at=season_unplayed["game_id"].map(first_seen).fillna(now)
    )
    for row in season_unplayed.itertuples():
        logger.info(f"Unplayed {row.status} game {row.game_id}, listed {row.listed_date}")

    others = existing[existing["end_year"] != end_year]
    frames = [frame for frame in (others, season_unplayed) if not frame.empty]
    combined = pd.concat(frames, ignore_index=True) if frames else season_unplayed
    atomic_write_csv(
        combined.reindex(columns=UNPLAYED_COLUMNS).sort_values(["end_year", "game_id"]), path
    )


def _write_manifest(manifest: dict, manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=manifest_path.parent, suffix=".tmp", delete=False, encoding="utf-8"
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(manifest, handle, indent=2, sort_keys=True)
        temp_path.replace(manifest_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
