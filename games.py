from utils import get_nba_season
from pandas import DataFrame
import pandas as pd
from teams_history import get_cities_conference


def read_games_file() -> DataFrame:
    file_path = "data/archive/Games.csv"
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


def filter_games(games, start_date, gameTypes) -> DataFrame:
    games = games.loc[games["gameType"].isin(gameTypes)]
    games = games.loc[games["gameDate"] >= start_date]
    games = games.loc[games["homeScore"] > 0]

    return games


def add_features_to_games(games):
    games["season"] = games["gameDate"].apply(get_nba_season)

    games["winnerTeam"] = games.apply(
        lambda x: str(x["hometeamName"])
        if x["winner"] == x["hometeamId"]
        else str(x["awayteamName"])
        if x["winner"] == x["awayteamId"]
        else "",
        axis=1,
    )

    games["gameDateOnlyStr"] = games["gameDate"].dt.strftime("%Y-%m-%d")

    games["pts_diff"] = games.apply(
        lambda x: x["homeScore"] - x["awayScore"],
        axis=1,
    )

    games["winner_home_bool"] = games.apply(
        lambda x: 1 if x["winner"] == x["hometeamId"] else 0, axis=1
    )

    games["winner_away_bool"] = games.apply(
        lambda x: 1 if x["winner"] != x["hometeamId"] else 0, axis=1
    )

    cities_conferences = get_cities_conference()

    games = games.merge(
        cities_conferences,
        how="left",
        left_on="hometeamCity",
        right_on="teamCity",
    ).drop(columns=["teamCity"])
    games.rename(columns={"Conference": "hometeamConference"}, inplace=True)

    games = games.merge(
        cities_conferences,
        how="left",
        left_on="awayteamCity",
        right_on="teamCity",
    ).drop(columns=["teamCity"])
    games.rename(columns={"Conference": "awayteamConference"}, inplace=True)

    return games
