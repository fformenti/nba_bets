"""Raw games data parsing and filtering.

``ingest_raw_games`` folds the raw historical archive into the one authoritative
history table. It used to write a separate seed CSV that only a first run ever
read, which meant `make full-rebuild` reparsed the raw archive and then silently
discarded it — the seed was never merged once the history table existed.
"""

from pathlib import Path

from pandas import DataFrame
import pandas as pd

from src.config.paths import (
    RAW_GAMES_PATH,
    INGESTED_GAMES_UPDATED_HISTORY_PATH,
)
from src.etl.utils.common import (
    CANONICAL_INGESTED_COLUMNS,
    assert_unique_game_ids,
    atomic_write_csv,
    coerce_numeric_columns,
    enrich_games_locations,
    get_nba_season,
)
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)
setup_logging(level="INFO")


def parse_raw_games(df: DataFrame) -> DataFrame:
    """Parse raw games DataFrame with proper data types and cleaning."""
    # Ensure gameDate is datetime if it isn't already
    if not pd.api.types.is_datetime64_any_dtype(df["gameDate"]):
        df["gameDate"] = pd.to_datetime(df["gameDate"], errors="coerce")

    # Convert specific columns to appropriate data types
    coerce_numeric_columns(df, ["gameId", "hometeamId", "awayteamId", "winner"])
    # arenaId may have NaN values, so use nullable integer type
    coerce_numeric_columns(df, ["arenaId"], dtype="Int64")
    df["seriesGameNumber"] = df["seriesGameNumber"].astype("float64")

    # Rename column
    df.rename(
        columns={"hometeamCity": "hometeamPrename", "awayteamCity": "awayteamPrename"},
        inplace=True,
    )

    # Handle empty strings in numeric columns (convert to NaN)
    numeric_cols = ["homeScore", "awayScore", "attendance"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # remove rows where gameDate is na
    df = df[df["gameDate"].notna()].copy()
    df["gameDateOnlyStr"] = df["gameDate"].dt.strftime("%Y-%m-%d")

    # add season column
    df["season"] = df["gameDate"].apply(get_nba_season)

    # Normalize city names and enrich with location columns
    df = enrich_games_locations(df)

    # add postponed column
    df["postponed"] = 0
    # Parenthesised deliberately: `|` binds tighter than `<=`, so writing this as
    # `awayScore <= 0 | homeScore.isna() | awayScore.isna()` compares the score
    # against a *boolean* and quietly stops flagging rows with a missing score —
    # which is exactly what an in-progress or suspended game looks like.
    mask_postponed = (
        (df["homeScore"] <= 0)
        | (df["awayScore"] <= 0)
        | df["homeScore"].isna()
        | df["awayScore"].isna()
    )
    df.loc[mask_postponed, "postponed"] = 1

    # add overtimes column
    df["overtimes"] = 0
    return df


def _played(df: DataFrame) -> pd.Series:
    """Whether each row records a game that actually happened."""
    return pd.to_numeric(df["postponed"], errors="coerce").fillna(0) == 0


def merge_into_history(parsed_games: DataFrame, history: DataFrame) -> DataFrame:
    """Union the freshly parsed raw archive onto the existing history table.

    The raw archive wins any gameId both hold: it is the curated upstream dump
    and carries ``attendance``, ``arenaId`` and ``seriesGameNumber``, which rows
    built from a live results payload do not have. History keeps every gameId the
    archive has not caught up with yet — that is what makes a rebuild safe to run
    against a table holding games collected since the archive was last published.

    One exception, and it is not cosmetic: a postponed row in the archive never
    supersedes a played row in history. The archive lags the league, so a game
    postponed and later replayed is still filed as the original scoreless fixture
    there long after the live feed has recorded the result. Letting raw win those
    would walk a played game back to postponed and drop it from the regular-season
    table — 22500529 did exactly that.
    """
    if history.empty:
        combined = parsed_games
    else:
        raw_ids = pd.to_numeric(parsed_games["gameId"], errors="coerce")
        history_ids = pd.to_numeric(history["gameId"], errors="coerce")

        stale_raw_ids = set(raw_ids[~_played(parsed_games)].dropna())
        protected = history_ids.isin(stale_raw_ids) & _played(history)
        if protected.any():
            logger.info(
                "Keeping %d played history row(s) the archive still lists as postponed: %s",
                protected.sum(),
                sorted(int(game_id) for game_id in history_ids[protected].dropna()),
            )
            parsed_games = parsed_games[~raw_ids.isin(set(history_ids[protected]))]
            raw_ids = pd.to_numeric(parsed_games["gameId"], errors="coerce")

        superseded = history_ids.isin(set(raw_ids.dropna()))
        if superseded.any():
            logger.info(
                "Raw archive supersedes %d row(s) already in history", superseded.sum()
            )
        kept = history[~superseded]
        logger.info("Keeping %d history row(s) not present in the raw archive", len(kept))
        frames = [df for df in [parsed_games, kept] if not df.empty]
        combined = pd.concat(frames, ignore_index=True)

    combined = combined.reindex(columns=CANONICAL_INGESTED_COLUMNS)
    combined = combined.sort_values(
        ["gameDate", "gameId"], na_position="last"
    ).reset_index(drop=True)
    assert_unique_game_ids(combined, "the raw archive merge")
    return combined


def ingest_raw_games(output_path: Path | None = None) -> None:
    """Parse the raw Games.csv and merge it into the history table."""
    if output_path is None:
        output_path = INGESTED_GAMES_UPDATED_HISTORY_PATH

    raw_games = pd.read_csv(RAW_GAMES_PATH, parse_dates=["gameDate"], low_memory=False)
    parsed_games = parse_raw_games(raw_games)
    logger.info(f"Parsed {len(parsed_games)} games from {RAW_GAMES_PATH}")

    if output_path.exists():
        history = pd.read_csv(output_path, parse_dates=["gameDate"], low_memory=False)
        logger.info(f"Loaded {len(history)} games from {output_path}")
    else:
        logger.info(f"No history table at {output_path} yet; creating it")
        history = pd.DataFrame()

    combined = merge_into_history(parsed_games, history)
    # Atomic: this is the authoritative history table, and nothing downstream can
    # tell a truncated table from a genuinely short one.
    atomic_write_csv(combined, output_path)
    logger.info(f"Saved {len(combined)} games to {output_path}")
