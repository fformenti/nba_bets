from utils import get_nba_season
from pandas import DataFrame
import pandas as pd


def read_games_file() -> DataFrame:
    # Read the CSV file directly into a pandas DataFrame
    file_path = "data/archive/Games.csv"  # Change to your actual filename
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


def filter_games(games, start_date, gameType) -> DataFrame:
    games = games.loc[games["gameType"] <= gameType]
    games = games.loc[games["gameDate"] >= start_date]
    games = games.loc[games["homeScore"] > 0]

    return games


def add_features_to_games(df):
    df["season"] = df["gameDate"].apply(get_nba_season)

    df["winnerTeam"] = df.apply(
        lambda x: str(x["hometeamName"])
        if x["winner"] == x["hometeamId"]
        else str(x["awayteamName"])
        if x["winner"] == x["awayteamId"]
        else "",
        axis=1,
    )

    df["gameDateOnlyStr"] = df["gameDate"].dt.strftime("%Y-%m-%d")

    df["pts_diff"] = df.apply(
        lambda x: x["homeScore"] - x["awayScore"],
        axis=1,
    )

    df["winner_home_bool"] = df.apply(
        lambda x: 1 if x["winner"] == x["hometeamId"] else 0, axis=1
    )

    df["winner_away_bool"] = df.apply(
        lambda x: 1 if x["winner"] != x["hometeamId"] else 0, axis=1
    )

    return df


def calculate_record(games):
    games = games.sort_values(["teamId", "gameDate"])

    games["total_wins"] = games.groupby("teamId")["win_bool"].transform(
        pd.Series.cumsum
    )

    # games["aux"] = 1
    # games["record"] = games["total_wins"] / games["games_count"]
    # games["games_count"] = games.groupby("teamId")["aux"].transform(pd.Series.cumsum)

    games["record"] = (
        games.groupby("teamId")["win_bool"]
        .expanding()  # Uses all available prior rows in the group
        .mean()
        .reset_index(level=0, drop=True)  # Align with original DataFrame
    )

    games["record_L5"] = (
        games.groupby("teamId")["win_bool"]
        .rolling(window=5, min_periods=1)  # min_periods=1 to avoid NaN for small groups
        .mean()
        .reset_index(level=0, drop=True)
    )

    return games
