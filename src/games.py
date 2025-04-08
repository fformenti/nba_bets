from pandas import DataFrame
import pandas as pd


def read_games_file() -> DataFrame:
    file_path = "data/raw/Games.csv"
    df = pd.read_csv(file_path, parse_dates=["gameDate"])

    # Convert specific columns to appropriate data types
    df["gameId"] = df["gameId"].astype("int64")
    df["hometeamId"] = df["hometeamId"].astype("int64")
    df["awayteamId"] = df["awayteamId"].astype("int64")
    df["winner"] = df["winner"].astype("int64")
    df["arenaId"] = df["arenaId"].astype("int64")
    df["seriesGameNumber"] = df["seriesGameNumber"].astype("float64")

    # Handle empty strings in numeric columns (convert to NaN)
    numeric_cols = ["homeScore", "awayScore", "attendance"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def filter_regular_season_games(games, start_date) -> DataFrame:
    GAMETYPES = ["Regular Season", "NBA Emirates Cup"]
    games = games.loc[games["gameType"].isin(GAMETYPES)]
    games = games.loc[games["gameDate"] >= start_date]
    games = games.loc[games["homeScore"] > 0]

    return games
