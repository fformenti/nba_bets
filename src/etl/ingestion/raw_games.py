"""Raw games data parsing and filtering."""

from pandas import DataFrame
import pandas as pd
from pathlib import Path
import sys

# Add project root to path to allow imports
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import LOCAL_RAW_GAMES_PATH, LOCAL_REGULAR_SEASON_GAMES_PATH


def get_nba_season(game_date):
    """Determine the NBA season for a given date.
    NBA season spans October-June, labeled as YYYY/YY where October is in the first year.
    Example: Feb 25, 2025 → season "2024/25" (because season started Oct 2024)"""
    year = game_date.year
    month = game_date.month

    # If month is October or later, season is current_year/next_year
    if month >= 9:
        return f"{year}/{(year + 1) % 100:02d}"
    # Otherwise season is previous_year/current_year
    else:
        return f"{year - 1}/{year % 100:02d}"


def parse_raw_games(df: DataFrame) -> DataFrame:
    """Parse raw games DataFrame with proper data types and cleaning."""
    # Ensure gameDate is datetime if it isn't already
    if not pd.api.types.is_datetime64_any_dtype(df["gameDate"]):
        df["gameDate"] = pd.to_datetime(df["gameDate"], errors="coerce")

    # Convert specific columns to appropriate data types
    # Use pd.to_numeric with errors='coerce' to handle empty strings/NaN before converting to int
    df["gameId"] = pd.to_numeric(df["gameId"], errors="coerce").astype("int64")
    df["hometeamId"] = pd.to_numeric(df["hometeamId"], errors="coerce").astype("int64")
    df["awayteamId"] = pd.to_numeric(df["awayteamId"], errors="coerce").astype("int64")
    df["winner"] = pd.to_numeric(df["winner"], errors="coerce").astype("int64")
    # arenaId may have NaN values, so use nullable integer type
    df["arenaId"] = pd.to_numeric(df["arenaId"], errors="coerce").astype("Int64")
    df["seriesGameNumber"] = df["seriesGameNumber"].astype("float64")

    # Handle empty strings in numeric columns (convert to NaN)
    numeric_cols = ["homeScore", "awayScore", "attendance"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter out invalid rows (combine conditions and use .copy() to preserve dtypes)
    mask = (
        df["gameDate"].notna()
        & df["homeScore"].notna()
        & df["awayScore"].notna()
        & (df["homeScore"] > 0)
        & (df["awayScore"] > 0)
    )
    df = df.loc[mask].copy()

    # Add formatted date string
    df["gameDateOnlyStr"] = df["gameDate"].dt.strftime("%Y-%m-%d")
    return df


def filter_regular_season_games(games) -> DataFrame:
    """Filter games to only include regular season games."""
    GAMETYPES = ["Regular Season", "NBA Emirates Cup"]
    games = games.loc[games["gameType"].isin(GAMETYPES)]
    games = games.drop(columns=["gameSubLabel", "gameLabel", "seriesGameNumber"]).copy()

    return games


if __name__ == "__main__":
    raw_games = pd.read_csv(LOCAL_RAW_GAMES_PATH, parse_dates=["gameDate"])
    parsed_games = parse_raw_games(raw_games)
    parsed_games["season"] = parsed_games["gameDate"].apply(get_nba_season)
    regular_season_games = filter_regular_season_games(parsed_games)
    regular_season_games.to_csv(LOCAL_REGULAR_SEASON_GAMES_PATH, index=False)
