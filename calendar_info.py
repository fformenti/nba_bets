import pandas as pd
from pandas import DataFrame
from numpy import nan


def is_home(home_game, away_game):
    if home_game == 1:
        return 1
    elif away_game == 1:
        return 0
    else:
        return nan


def is_away(home_game, away_game):
    if home_game == 1:
        return 0
    elif away_game == 1:
        return 1
    else:
        return nan


def make_rested_days_table(
    games_filtered, start_date, end_date, teams_season
) -> DataFrame:
    date_range = pd.date_range(start=start_date, end=end_date)
    rested_days = pd.MultiIndex.from_product(
        [date_range, teams_season], names=["gameDate", "teamId"]
    ).to_frame(index=False)
    rested_days["gameDateOnlyStr"] = rested_days["gameDate"].dt.strftime("%Y-%m-%d")

    rested_days = rested_days.merge(
        games_filtered[["gameDateOnlyStr", "hometeamId"]],
        left_on=["gameDateOnlyStr", "teamId"],
        right_on=["gameDateOnlyStr", "hometeamId"],
        how="left",
        indicator="home_game",
    ).drop("hometeamId", axis=1)

    rested_days = rested_days.merge(
        games_filtered[["gameDateOnlyStr", "awayteamId"]],
        left_on=["gameDateOnlyStr", "teamId"],
        right_on=["gameDateOnlyStr", "awayteamId"],
        how="left",
        indicator="away_game",
    ).drop("awayteamId", axis=1)

    rested_days["home_game"] = rested_days["home_game"].apply(
        lambda x: 1 if x == "both" else 0
    )

    rested_days["away_game"] = rested_days["away_game"].apply(
        lambda x: 1 if x == "both" else 0
    )

    rested_days["rest"] = rested_days.apply(
        lambda x: abs((x["home_game"] | x["away_game"]) - 1), axis=1
    )

    rested_days = rested_days.sort_values(["teamId", "gameDate"])
    rested_days["rest"] = (
        rested_days.groupby("teamId")["rest"].shift(1).fillna(1).astype(int)
    )

    rested_days["rested_days"] = rested_days.groupby(
        ["teamId", (rested_days["rest"] == 0).cumsum()]
    )["rest"].transform("cumsum")

    # At Home Count
    rested_days["at_home_indicator"] = rested_days.apply(
        lambda x: is_home(x["home_game"], x["away_game"]), axis=1
    )
    rested_days["at_home_indicator"] = (
        rested_days.groupby("teamId")["at_home_indicator"].ffill().fillna(1).astype(int)
    )
    rested_days["days_at_home"] = rested_days.groupby(
        ["teamId", (rested_days["at_home_indicator"] == 0).cumsum()]
    )["at_home_indicator"].transform("cumsum")

    # On the Road Count
    rested_days["at_road_indicator"] = rested_days.apply(
        lambda x: is_away(x["home_game"], x["away_game"]), axis=1
    )
    rested_days["at_road_indicator"] = (
        rested_days.groupby("teamId")["at_road_indicator"].ffill().fillna(0).astype(int)
    )
    rested_days["days_on_road"] = rested_days.groupby(
        ["teamId", (rested_days["at_road_indicator"] == 0).cumsum()]
    )["at_road_indicator"].transform("cumsum")

    return rested_days
