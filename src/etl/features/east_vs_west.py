import pandas as pd

from src.etl.utils.common import get_season_date_range

# Season-to-date conference records over interconference games, one row per
# (season, date), shifted a day so a game never sees its own result.
#
# `location=None` averages the two venues out: east_record_adjusted is the mean
# of East's win rate hosting and East's win rate visiting. `location="East"` /
# `"West"` return exactly the venue component that average discards — how well a
# conference holds its own floor. The two are complements, and both reach the
# model (src/ml/features/engineering.py::create_conference_features).
#
# The games_played_* columns are divisors here and nothing downstream reads
# them, so they are not returned. `season` is returned because the frame is
# grouped by (season, date) and the aggregator has to join on both.


def make_east_west_record(games, location=None):
    east_west_record = games[games["hometeamConference"] != games["awayteamConference"]]

    if location == "East":
        east_west_record = east_west_record[
            east_west_record["hometeamConference"] == "East"
        ]
        suffix = "_at_east"
        return_cols = ["season", "gameDateOnlyStr", f"east_record{suffix}"]
    elif location == "West":
        east_west_record = east_west_record[
            east_west_record["hometeamConference"] == "West"
        ]
        suffix = "_at_west"
        return_cols = ["season", "gameDateOnlyStr", f"west_record{suffix}"]
    else:
        suffix = ""
        return_cols = [
            "season",
            "gameDateOnlyStr",
            "east_record_adjusted",
            "west_record_adjusted",
        ]

    east_west_record = east_west_record[
        [
            "gameDate",
            "gameDateOnlyStr",
            "season",
            "hometeamConference",
            "awayteamConference",
            "winnerteamConference",
        ]
    ]

    east_west_record["east_wins"] = east_west_record["winnerteamConference"].apply(
        lambda x: 1 if x == "East" else 0
    )

    east_west_record["west_wins"] = east_west_record["winnerteamConference"].apply(
        lambda x: 1 if x == "West" else 0
    )

    east_west_record["east_games"] = east_west_record["hometeamConference"].apply(
        lambda x: 1 if x == "East" else 0
    )

    east_west_record["west_games"] = east_west_record["hometeamConference"].apply(
        lambda x: 1 if x == "West" else 0
    )

    east_west_record["east_wins_at_east"] = (
        (east_west_record["winnerteamConference"] == "East")
        & (east_west_record["hometeamConference"] == "East")
    ).astype(int)

    east_west_record["east_wins_at_west"] = (
        (east_west_record["winnerteamConference"] == "East")
        & (east_west_record["hometeamConference"] == "West")
    ).astype(int)

    east_west_record_by_date = (
        east_west_record.groupby(["season", "gameDateOnlyStr"])
        .agg(
            east_wins_date=("east_wins", "sum"),
            west_wins_date=("west_wins", "sum"),
            games_played_date_at_east=("east_games", "sum"),
            games_played_date_at_west=("west_games", "sum"),
            east_wins_date_at_east=("east_wins_at_east", "sum"),
            east_wins_date_at_west=("east_wins_at_west", "sum"),
        )
        .reset_index()
    )

    all_season_date_range = get_season_date_range(games)

    # Join on season too: the right side is grouped by (season, date), so joining
    # on date alone would duplicate rows for any date shared by two seasons.
    east_west_record_by_date = all_season_date_range.merge(
        east_west_record_by_date[
            [
                "season",
                "gameDateOnlyStr",
                "east_wins_date",
                "west_wins_date",
                "games_played_date_at_east",
                "games_played_date_at_west",
                "east_wins_date_at_east",
                "east_wins_date_at_west",
            ]
        ],
        on=["season", "gameDateOnlyStr"],
        how="left",
        validate="1:1",
    )

    east_west_record_by_date.fillna(0, inplace=True)

    # Ensure dataframe is sorted by season and date (oldest to most recent) before cumsum
    east_west_record_by_date = east_west_record_by_date.sort_values(
        by=["season", "gameDateOnlyStr"]
    ).reset_index(drop=True)

    east_west_record_by_date["east_wins"] = (
        east_west_record_by_date.groupby("season")["east_wins_date"]
        .cumsum()
        .groupby(east_west_record_by_date["season"])
        .shift(1)
        .fillna(0)
    )

    east_west_record_by_date["west_wins"] = (
        east_west_record_by_date.groupby("season")["west_wins_date"]
        .cumsum()
        .groupby(east_west_record_by_date["season"])
        .shift(1)
        .fillna(0)
    )

    east_west_record_by_date["games_played_at_east"] = (
        east_west_record_by_date.groupby("season")["games_played_date_at_east"]
        .cumsum()
        .groupby(east_west_record_by_date["season"])
        .shift(1)
        .fillna(0)
    )

    east_west_record_by_date["games_played_at_west"] = (
        east_west_record_by_date.groupby("season")["games_played_date_at_west"]
        .cumsum()
        .groupby(east_west_record_by_date["season"])
        .shift(1)
        .fillna(0)
    )

    east_west_record_by_date["east_wins_at_east"] = (
        east_west_record_by_date.groupby("season")["east_wins_date_at_east"]
        .cumsum()
        .groupby(east_west_record_by_date["season"])
        .shift(1)
        .fillna(0)
    )

    east_west_record_by_date["east_wins_at_west"] = (
        east_west_record_by_date.groupby("season")["east_wins_date_at_west"]
        .cumsum()
        .groupby(east_west_record_by_date["season"])
        .shift(1)
        .fillna(0)
    )

    east_west_record_by_date.drop(
        columns=[
            "east_wins_date",
            "west_wins_date",
            "games_played_date_at_east",
            "games_played_date_at_west",
            "east_wins_date_at_east",
            "east_wins_date_at_west",
        ],
        inplace=True,
    )
    east_west_record_by_date["games_played_east_vs_west"] = (
        east_west_record_by_date["games_played_at_west"]
        + east_west_record_by_date["games_played_at_east"]
    )

    safe_gp = east_west_record_by_date["games_played_east_vs_west"].replace(0, float("nan"))
    east_west_record_by_date[f"east_record{suffix}"] = (
        east_west_record_by_date["east_wins"] / safe_gp
    ).fillna(0.0)
    east_west_record_by_date[f"west_record{suffix}"] = (
        east_west_record_by_date["west_wins"] / safe_gp
    ).fillna(0.0)

    if location is None:
        safe_gp_east = east_west_record_by_date["games_played_at_east"].replace(0, float("nan"))
        safe_gp_west = east_west_record_by_date["games_played_at_west"].replace(0, float("nan"))

        east_wr_at_east = east_west_record_by_date["east_wins_at_east"] / safe_gp_east
        east_wr_at_west = east_west_record_by_date["east_wins_at_west"] / safe_gp_west

        east_west_record_by_date["east_record_adjusted"] = (
            pd.DataFrame({"at_east": east_wr_at_east, "at_west": east_wr_at_west})
            .mean(axis=1)
            .fillna(0.0)
        )

        west_wr_at_east = (
            east_west_record_by_date["games_played_at_east"]
            - east_west_record_by_date["east_wins_at_east"]
        ) / safe_gp_east
        west_wr_at_west = (
            east_west_record_by_date["games_played_at_west"]
            - east_west_record_by_date["east_wins_at_west"]
        ) / safe_gp_west

        east_west_record_by_date["west_record_adjusted"] = (
            pd.DataFrame({"at_east": west_wr_at_east, "at_west": west_wr_at_west})
            .mean(axis=1)
            .fillna(0.0)
        )

    return east_west_record_by_date[return_cols]
