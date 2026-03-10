"""Raw games data parsing and filtering."""

from pandas import DataFrame
import pandas as pd

from src.config.paths import (
    RAW_GAMES_PATH,
    INGESTED_GAMES_PATH,
    TEAMS_CITIES_LOCATIONS_HISTORY_PROCESSED_PATH,
)
from src.etl.utils.common import coerce_numeric_columns, get_nba_season
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)
setup_logging(level="INFO")


def get_teams_locations(df: DataFrame) -> DataFrame:
    teams_cities_states = pd.read_csv(TEAMS_CITIES_LOCATIONS_HISTORY_PROCESSED_PATH)
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

    # Normalize
    normalize_citi_names = {"LA": "Los Angeles"}
    df["hometeamPrename"] = df["hometeamPrename"].replace(normalize_citi_names)
    df["awayteamPrename"] = df["awayteamPrename"].replace(normalize_citi_names)

    # Handle empty strings in numeric columns (convert to NaN)
    numeric_cols = ["homeScore", "awayScore", "attendance"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # remove rows where gameDate is na
    df = df[df["gameDate"].notna()].copy()
    df["gameDateOnlyStr"] = df["gameDate"].dt.strftime("%Y-%m-%d")

    # add season column
    df["season"] = df["gameDate"].apply(get_nba_season)

    # Get real city and state from teamId and season
    df = get_teams_locations(df)

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
