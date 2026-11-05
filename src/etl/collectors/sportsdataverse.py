"""Download season files from the sportsdataverse data releases on GitHub.

Source, with no key and no scraping:

    https://github.com/sportsdataverse/sportsdataverse-data/releases/download/<tag>/<file>

The releases republish stats.nba.com data, one file per season keyed by the
season's END year (1997 is our "1996/97"). Each release has one
``timestamp.txt`` for the whole tag, not one per season. What each file holds,
and the terms that come with it, are in the modules that use it:
``boxscores.py`` and ``results/release_source.py``. Research:
docs/findings/pipeline/2026-11-02-nba-data-sources.md.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config.constants import SCHEDULE_TIMEZONE
from src.etl.utils.common import get_nba_season

RELEASE_URL = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
# 1996/97, the first season in the releases and min_season in configs/train/_defaults.yaml.
FIRST_SEASON_END_YEAR = 1997

# (connect, read) seconds, on every request.
REQUEST_TIMEOUT = (10, 60)
USER_AGENT = "nba_bets data fetcher (python-requests)"


class SchemaDriftError(RuntimeError):
    """A release file no longer has the columns this code was written against."""


@dataclass(frozen=True)
class ReleaseFile:
    """One release's per-season file, its frozen schema, and where it is stored.

    ``pattern`` names the file for a season, e.g. ``"nba_schedule_{year}.parquet"``.
    ``directory`` is ``None`` for a file that is read but never stored.
    """

    name: str
    tag: str
    pattern: str
    schema: dict[str, str]
    directory: Path | None = None

    def filename(self, end_year: int) -> str:
        return self.pattern.format(year=end_year)

    def path(self, end_year: int) -> Path:
        if self.directory is None:
            raise ValueError(f"{self.name} files are not stored")
        return self.directory / self.filename(end_year)

    def url(self, end_year: int) -> str:
        return f"{RELEASE_URL}/{self.tag}/{self.filename(end_year)}"

    @property
    def timestamp_url(self) -> str:
        return f"{RELEASE_URL}/{self.tag}/timestamp.txt"


def season_label(end_year: int) -> str:
    """``1997`` → ``"1996/97"``, the ``season`` format of our game history."""
    return f"{end_year - 1}/{end_year % 100:02d}"


def season_end_year(game_id: str | int) -> int:
    """End year of the season a gameId belongs to: ``0022400524`` → 2025.

    Digits 4-5 of the padded id are the season's start year; the league's ids
    start in 1946, so ``46``-``99`` are 1900s.
    """
    start = int(f"{int(game_id):010d}"[3:5])
    return (1900 if start >= 46 else 2000) + start + 1


def today_eastern() -> date:
    """Today on the league's calendar. Game dates are Eastern dates, and a UTC
    date runs a day ahead every evening, while games are still being played."""
    return pd.Timestamp.now(tz=SCHEDULE_TIMEZONE).date()


def current_season_end_year(today: date | None = None) -> int:
    """End year of the season ``today`` falls in, by the history's own rule."""
    label = get_nba_season(pd.Timestamp(today or today_eastern()))
    return int(label[:4]) + 1


def all_seasons(today: date | None = None) -> list[int]:
    return list(range(FIRST_SEASON_END_YEAR, current_season_end_year(today) + 1))


def make_session() -> requests.Session:
    """Retries network errors and 5xx with exponential backoff; GET only."""
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods={"GET"},
    )
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def get(session: requests.Session, url: str) -> bytes:
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.content


def release_timestamp(session: requests.Session, release: ReleaseFile) -> str:
    return get(session, release.timestamp_url).decode().strip()


def download_table(session: requests.Session, release: ReleaseFile, end_year: int) -> pa.Table:
    """One season's file, schema-checked. Raises ``requests.HTTPError`` on a 404."""
    table = pq.read_table(io.BytesIO(get(session, release.url(end_year))))
    check_schema(table, release, end_year)
    return table


def check_schema(table: pa.Table, release: ReleaseFile, end_year: int) -> None:
    """Raise on any added, missing, reordered or retyped column, or an empty file."""
    found = {field.name: _type_name(field.type) for field in table.schema}
    if found != release.schema or list(found) != list(release.schema):
        missing = sorted(set(release.schema) - set(found))
        added = sorted(set(found) - set(release.schema))
        retyped = {
            name: f"{release.schema[name]} -> {found[name]}"
            for name in release.schema.keys() & found.keys()
            if release.schema[name] != found[name]
        }
        raise SchemaDriftError(
            f"{release.filename(end_year)} schema changed: missing={missing} added={added} "
            f"retyped={retyped} order_changed={not (missing or added or retyped)}"
        )
    if table.num_rows == 0:
        raise SchemaDriftError(f"{release.filename(end_year)} is empty")
    seasons = set(table.column("season").unique().to_pylist())
    if seasons != {end_year}:
        raise SchemaDriftError(
            f"{release.filename(end_year)} holds seasons {sorted(seasons)}, expected {end_year}"
        )


def _type_name(arrow_type: pa.DataType) -> str:
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return "string"
    return str(arrow_type)
