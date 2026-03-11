"""Raw games data parsing and filtering."""

from pandas import DataFrame
import pandas as pd

from src.config.paths import (
    RAW_GAMES_PATH,
    INGESTED_GAMES_PATH,
)
from src.etl.utils.common import coerce_numeric_columns, enrich_games_locations, get_nba_season
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
    mask_postponed = (df["homeScore"] <= 0) | (
        df["awayScore"] <= 0 | df["homeScore"].isna() | df["awayScore"].isna()
    )
    df.loc[mask_postponed, "postponed"] = 1

    # add overtimes column
    df["overtimes"] = 0
    return df


def main():
    raw_games = pd.read_csv(RAW_GAMES_PATH, parse_dates=["gameDate"], low_memory=False)
    parsed_games = parse_raw_games(raw_games)
    parsed_games.to_csv(INGESTED_GAMES_PATH, index=False)
    logger.info(f"Saved {len(parsed_games)} parsed games to {INGESTED_GAMES_PATH}")


if __name__ == "__main__":
    main()
