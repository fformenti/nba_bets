"""Common utility functions for data processing."""

import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.constants import (
    CSV_FLOAT_FORMAT,
    GAME_ID_TYPE_PREFIX,
    NEUTRAL_COURT_GAME_LABELS,
)
from src.config.paths import TEAMS_LOCATIONS_REFERENCE_PATH


def require_reference_file(path: Path, make_target: str) -> None:
    """Fail early, and legibly, when a reference input is missing.

    These files are generated once by an external service and cannot be rebuilt
    from anything in this repo (see ``RAW_REFERENCE_DIR``). Without this check a
    missing one surfaces as a bare ``FileNotFoundError`` from inside pandas,
    several frames below the step the user actually ran.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Reference input not found: {path}\n"
            f"It is an input, not a build artifact — nothing in the pipeline can "
            f"regenerate it offline. Restore it from a backup, or rebuild it with "
            f"`make {make_target}` (needs API credentials; see docs/PIPELINE_AUDIT.md)."
        )


def game_type_from_id(game_id: Any) -> str | None:
    """The kind of game a gameId encodes, or ``None`` if it cannot be read.

    The third digit of a zero-padded NBA gameId is the game type: ``0022500798``
    is a regular-season game, ``0042500101`` a playoff game. Incrementally
    collected games have no other source of truth for this — results payloads
    carry no ``gameType``, and the league schedule's ``gameLabel`` is blank for
    every playoff round — so a game fetched in April would otherwise be filed as
    regular season.

    Returns ``None`` rather than guessing, so callers can decide whether an
    unreadable id is fatal.
    """
    if game_id is None or (isinstance(game_id, float) and pd.isna(game_id)):
        return None
    try:
        padded = f"{int(game_id):010d}"
    except (TypeError, ValueError):
        return None
    if len(padded) != 10:
        return None
    return GAME_ID_TYPE_PREFIX.get(padded[2])


def read_json(file_path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON file.

    Shared because three modules had grown their own copy of this.
    """
    with Path(file_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Write via a temp file in the same directory, then rename over the target.

    Use this for any file the rest of the pipeline reads back as authoritative.
    A plain ``to_csv`` that is interrupted — Ctrl-C, a full disk — leaves a
    truncated file behind, and nothing downstream can tell a truncated table
    from a genuinely short one.

    The temp file goes in the target's own directory so the final ``replace`` is
    a same-filesystem rename, which is atomic.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=output_path.parent, suffix=".tmp", delete=False, encoding="utf-8"
        ) as handle:
            temp_path = Path(handle.name)
            df.to_csv(handle, index=False)
        temp_path.replace(output_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def save_feature_table(df: pd.DataFrame, path, **kwargs) -> None:
    """
    Write a feature table to CSV, rounding floats to CSV_FLOAT_FORMAT.

    This is the only place feature values get rounded.  Every calculation
    upstream runs at full double precision so that intermediates feeding
    other features are never truncated.
    """
    df.to_csv(path, index=False, float_format=CSV_FLOAT_FORMAT, **kwargs)


CANONICAL_INGESTED_COLUMNS = [
    "gameId",
    "gameDate",
    "gameDateOnlyStr",
    "season",
    "hometeamPrename",
    "hometeamName",
    "hometeamId",
    "awayteamPrename",
    "awayteamName",
    "awayteamId",
    "homeScore",
    "awayScore",
    "winner",
    "overtimes",
    "postponed",
    "gameType",
    "attendance",
    "arenaId",
    "gameLabel",
    "gameSubLabel",
    "seriesGameNumber",
    "hometeamLocation",
    "gameLocation",
    "awayteamLocation",
]


def get_teams_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich a games DataFrame with home/away/game location columns via teams history lookup."""
    require_reference_file(TEAMS_LOCATIONS_REFERENCE_PATH, "build-teams-locations")
    teams_cities_states = pd.read_csv(TEAMS_LOCATIONS_REFERENCE_PATH)
    # Home team location
    df = df.merge(
        teams_cities_states[["teamId", "season", "city", "state"]],
        left_on=["hometeamId", "season"],
        right_on=["teamId", "season"],
        how="left",
    ).drop(columns=["teamId"])
    df["hometeamLocation"] = df["city"] + ", " + df["state"]
    df.drop(columns=["city", "state"], inplace=True)

    # Game Location (To do: in the future it should reference arenaId)
    df["gameLocation"] = df["hometeamLocation"]

    # Away team location
    df = df.merge(
        teams_cities_states[["teamId", "season", "city", "state"]],
        left_on=["awayteamId", "season"],
        right_on=["teamId", "season"],
        how="left",
    ).drop(columns=["teamId"])
    df["awayteamLocation"] = df["city"] + ", " + df["state"]
    df.drop(columns=["city", "state"], inplace=True)
    return df


def enrich_games_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize city names and add location columns if missing or incomplete."""
    normalize_city_names = {"LA": "Los Angeles"}
    if "hometeamPrename" in df.columns:
        df["hometeamPrename"] = df["hometeamPrename"].replace(normalize_city_names)
    if "awayteamPrename" in df.columns:
        df["awayteamPrename"] = df["awayteamPrename"].replace(normalize_city_names)

    location_cols = ["hometeamLocation", "awayteamLocation", "gameLocation"]
    needs_enrichment = any(col not in df.columns for col in location_cols)
    if not needs_enrichment:
        needs_enrichment = df[location_cols].isna().any().any()

    if needs_enrichment:
        # Drop any partial location columns before re-merging
        existing_loc_cols = [c for c in location_cols if c in df.columns]
        if existing_loc_cols:
            df = df.drop(columns=existing_loc_cols)
        df = get_teams_locations(df)

    return df


def has_result(games: pd.DataFrame) -> pd.Series:
    """Which rows are games that have actually been played.

    ``winner`` holds the winning teamId, so a played game's winner is one of its
    two teams. The prediction pipeline concatenates the upcoming slate onto
    history and builds the feature tables from the combined frame — those slate
    rows are placeholders from ``fix_upcoming_games_cols`` with ``winner = 0``
    and ``0-0`` scores.

    A placeholder must still *receive* features (that is the point of predicting
    it), but it must never *contribute* to anything cumulative: read as a game
    outcome, ``winner = 0`` is a loss for both sides, which quietly depressed
    every opponent win percentage the strength-of-schedule lookup served to the
    rest of the slate. Builders that aggregate across games filter on this.
    """
    winner = pd.to_numeric(games["winner"], errors="coerce")
    home = pd.to_numeric(games["hometeamId"], errors="coerce")
    away = pd.to_numeric(games["awayteamId"], errors="coerce")
    return (winner == home) | (winner == away)


def assert_unique_game_ids(df: pd.DataFrame, context: str) -> None:
    """Raise if the frame holds the same gameId twice.

    The history table is the one place a duplicate fixture is unrecoverable —
    every feature table downstream counts games — so both writers check before
    the write rather than after.
    """
    duplicated = pd.to_numeric(df["gameId"], errors="coerce").dropna().duplicated()
    if duplicated.any():
        raise ValueError(
            f"Refusing to write: {duplicated.sum()} duplicate gameId(s) after {context}."
        )


def deduplicate_games(
    existing_df: pd.DataFrame, new_df: pd.DataFrame
) -> pd.DataFrame:
    """Remove rows from existing_df that new_df supersedes.

    Matching is by ``gameId``, which is stable when a game is rescheduled. The
    old composite key ``(gameDateOnlyStr, hometeamId, awayteamId)`` could not do
    this: a postponed game replayed on another date changed its date component,
    so the original fixture survived alongside the replay and the same gameId
    appeared twice.

    Rows missing a gameId fall back to the composite key, which is how the
    oldest historical rows are matched. That fallback still collapses a same-day
    home-and-home between two teams down to a single game.
    """
    composite_cols = ["gameDateOnlyStr", "hometeamId", "awayteamId"]

    # A first run has no history table at all, so existing_df arrives with no
    # columns and the composite-key lookup below would raise on all three.
    if existing_df.empty:
        return existing_df

    def _ids(df: pd.DataFrame) -> pd.Series:
        if "gameId" not in df.columns:
            return pd.Series(pd.NA, index=df.index, dtype="Float64")
        return pd.to_numeric(df["gameId"], errors="coerce")

    def _composite_keys(df: pd.DataFrame) -> pd.DataFrame:
        # Normalize dtypes before comparing: an id read as float64 or Int64 would
        # never match the same id as int64 in a set lookup, silently skipping
        # the dedupe and duplicating games.
        normalized = df[composite_cols].copy()
        for col in ["hometeamId", "awayteamId"]:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
        normalized["gameDateOnlyStr"] = normalized["gameDateOnlyStr"].astype("string")
        return normalized

    new_ids = set(_ids(new_df).dropna().astype("int64"))
    new_composite = set(
        _composite_keys(new_df).dropna().itertuples(index=False, name=None)
    )

    existing_ids = _ids(existing_df)
    existing_composite = _composite_keys(existing_df)

    # Build the mask over the FULL frame, not a dropna'd copy: a shorter mask
    # raises "Boolean index has wrong length" against existing_df.
    mask = []
    for game_id, composite in zip(
        existing_ids, existing_composite.itertuples(index=False, name=None)
    ):
        if pd.notna(game_id):
            mask.append(int(game_id) not in new_ids)
        else:
            mask.append(composite not in new_composite)
    return existing_df.loc[mask].copy()


def coerce_numeric_columns(
    df: pd.DataFrame, columns: list[str], dtype: str = "int64"
) -> pd.DataFrame:
    """Convert columns to numeric types, coercing errors to NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (modified in place).
    columns : list[str]
        Column names to convert.
    dtype : str, default="int64"
        Target dtype after conversion. Use "Int64" for nullable integers.

    Raises
    ------
    ValueError
        If a value cannot be parsed and *dtype* cannot hold NaN.
    """
    for col in columns:
        if col not in df.columns:
            continue

        converted = pd.to_numeric(df[col], errors="coerce")

        # The default int64 cannot hold NaN — whether pre-existing or just
        # produced by errors="coerce". Report which values are at fault instead
        # of letting pandas raise an opaque "Cannot convert non-finite values".
        if dtype.startswith("int") and converted.isna().any():
            unparseable = converted.isna() & df[col].notna()
            sample = df.loc[unparseable, col].unique()[:5].tolist()
            raise ValueError(
                f"Column {col!r} has {int(converted.isna().sum())} value(s) that "
                f"cannot be represented as {dtype} "
                f"({int(unparseable.sum())} unparseable, e.g. {sample}; "
                f"the rest were already null). "
                f'Pass dtype="Int64" to allow nulls.'
            )

        df[col] = converted.astype(dtype)
    return df


def get_season_date_range(games):
    all_season_date_range = []
    seasons = games["season"].unique()
    for season in seasons:
        games_season = games.loc[games["season"] == season].copy()
        season_start = games_season["gameDate"].min()
        season_end = games_season["gameDate"].max()

        # Normalize to midnight: pd.date_range carries the start's time-of-day, so a
        # finale tipping off earlier in the day than the opener would fall past
        # `end` and drop the season's last day entirely.
        date_range = pd.date_range(
            start=season_start.normalize(), end=season_end.normalize()
        )
        season_dates = pd.MultiIndex.from_product(
            [date_range], names=["gameDate"]
        ).to_frame(index=False)
        season_dates["season"] = season
        season_dates["gameDateOnlyStr"] = season_dates["gameDate"].dt.strftime(
            "%Y-%m-%d"
        )
        season_dates.drop(columns=["gameDate"], inplace=True)
        all_season_date_range.append(season_dates)

    return pd.concat(all_season_date_range, ignore_index=True)


def get_nba_season(game_date):
    """
    Determine the NBA season for a given date.

    NBA season spans October-June, labeled as YYYY/YY where October is in the first year.
    Example: Feb 25, 2025 → season "2024/25" (because season started Oct 2024)

    Parameters
    ----------
    game_date : pd.Timestamp or datetime
        Game date

    Returns
    -------
    str
        Season string in format "YYYY/YY"
    """
    year = game_date.year
    month = game_date.month

    # If month is October or later, season is current_year/next_year
    if month >= 9:
        return f"{year}/{(year + 1) % 100:02d}"
    # Otherwise season is previous_year/current_year
    else:
        return f"{year - 1}/{year % 100:02d}"



def add_neutral_court_game_flag(
    df: pd.DataFrame,
    game_label_column: str = "gameLabel",
    drop_label_column: bool = True,
) -> pd.DataFrame:
    """
    Add a boolean flag indicating if a game is played on a neutral court.

    Neutral court games include international games and Las Vegas games
    as defined in NEUTRAL_COURT_GAME_LABELS.

    Parameters
    ----------
    df : pd.DataFrame
        Games DataFrame with a game label column
    game_label_column : str, default="gameLabel"
        Name of the column containing game labels
    drop_label_column : bool, default=True
        Whether to drop the game label column after processing

    Returns
    -------
    pd.DataFrame
        DataFrame with 'is_neutral_court_game' boolean column added.
        If game_label_column doesn't exist, all values will be False.
    """
    df = df.copy()

    df["is_neutral_court_game"] = df[game_label_column].isin(NEUTRAL_COURT_GAME_LABELS)

    if drop_label_column:
        df = df.drop(columns=[game_label_column]).copy()

    return df
