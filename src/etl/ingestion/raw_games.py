"""Raw games data parsing and filtering."""

from pandas import DataFrame
import pandas as pd

from src.config.paths import (
    RAW_GAMES_PATH,
    REGULAR_SEASON_GAMES_PATH,
    NON_POSITIVE_SCORE_PATH,
)
from src.etl.utils.common import get_nba_season, add_neutral_court_game_flag


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

    mask_non_positive_scores = (df["homeScore"] <= 0) | (df["awayScore"] <= 0)

    df = df.loc[mask].copy()
    df_non_positive_score = df.loc[mask_non_positive_scores].copy()

    # Add formatted date string
    df["gameDateOnlyStr"] = df["gameDate"].dt.strftime("%Y-%m-%d")
    return df, df_non_positive_score


def filter_regular_season_games(games) -> DataFrame:
    """Filter games to only include regular season games."""

    # Changes were made to the raw games season 2025/26 has a different patttern from previous years
    not_gametype_regular_season = ["Playoffs", "Preseason", "Play-in Tournament"]
    games = games.loc[~games["gameType"].isin(not_gametype_regular_season)]
    gamelabel_preseason = ["Preseason"]
    games = games.loc[~games["gameType"].isin(gamelabel_preseason)]
    games = games.drop(columns=["gameSubLabel", "seriesGameNumber"]).copy()

    return games


def build_regular_season_games(raw_games: DataFrame) -> DataFrame:
    """Parse raw games, add season, and filter to regular season games."""
    parsed_games, df_non_positive_score = parse_raw_games(raw_games)
    # save df_non_positive_score to csv
    df_non_positive_score.to_csv(NON_POSITIVE_SCORE_PATH, index=False)
    parsed_games = add_neutral_court_game_flag(
        parsed_games, game_label_column="gameLabel", drop_label_column=True
    )
    parsed_games["season"] = parsed_games["gameDate"].apply(get_nba_season)
    return filter_regular_season_games(parsed_games)


if __name__ == "__main__":
    raw_games = pd.read_csv(RAW_GAMES_PATH, parse_dates=["gameDate"])
    regular_season_games = build_regular_season_games(raw_games)
    regular_season_games.to_csv(REGULAR_SEASON_GAMES_PATH, index=False)
