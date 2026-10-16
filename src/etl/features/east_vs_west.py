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
# `games_played_east_vs_west` (returned only for `location=None`) is how far
# into the season the sample behind `east_record_adjusted`/`west_record_adjusted`
# actually is. create_conference_features uses it to blend those values with
# last season's final record (make_last_season_east_west_record, below) early
# in a season, when the current-season sample is thin.


def _cross_conference_indicators(games, location=None):
    """Interconference games, filtered by host conference and tagged with the
    per-game win/host indicator columns both builders below aggregate.

    Shared so the current-season cumulative builder (``make_east_west_record``)
    and the prior-season final builder (``make_last_season_east_west_record``)
    stay numerically consistent without duplicating the indicator math.

    Returns
    -------
    (pd.DataFrame, str)
        The filtered frame — gameDate, gameDateOnlyStr, season, the conference
        columns, plus the six indicator columns — and the location suffix
        ("", "_at_east" or "_at_west") used to name the ratio column.
    """
    east_west_record = games[games["hometeamConference"] != games["awayteamConference"]]

    if location == "East":
        east_west_record = east_west_record[
            east_west_record["hometeamConference"] == "East"
        ]
        suffix = "_at_east"
    elif location == "West":
        east_west_record = east_west_record[
            east_west_record["hometeamConference"] == "West"
        ]
        suffix = "_at_west"
    else:
        suffix = ""

    east_west_record = east_west_record[
        [
            "gameDate",
            "gameDateOnlyStr",
            "season",
            "hometeamConference",
            "awayteamConference",
            "winnerteamConference",
        ]
    ].copy()

    east_west_record["east_wins"] = (
        east_west_record["winnerteamConference"] == "East"
    ).astype(int)
    east_west_record["west_wins"] = (
        east_west_record["winnerteamConference"] == "West"
    ).astype(int)
    east_west_record["east_games"] = (
        east_west_record["hometeamConference"] == "East"
    ).astype(int)
    east_west_record["west_games"] = (
        east_west_record["hometeamConference"] == "West"
    ).astype(int)
    east_west_record["east_wins_at_east"] = (
        (east_west_record["winnerteamConference"] == "East")
        & (east_west_record["hometeamConference"] == "East")
    ).astype(int)
    east_west_record["east_wins_at_west"] = (
        (east_west_record["winnerteamConference"] == "East")
        & (east_west_record["hometeamConference"] == "West")
    ).astype(int)

    return east_west_record, suffix


def make_east_west_record(games, location=None):
    east_west_record, suffix = _cross_conference_indicators(games, location=location)

    if location == "East":
        return_cols = ["season", "gameDateOnlyStr", f"east_record{suffix}"]
    elif location == "West":
        return_cols = ["season", "gameDateOnlyStr", f"west_record{suffix}"]
    else:
        return_cols = [
            "season",
            "gameDateOnlyStr",
            "east_record_adjusted",
            "west_record_adjusted",
            "games_played_east_vs_west",
        ]

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


def make_last_season_east_west_record(games, location=None):
    """Each season's *final* conference-matchup record, one row per season.

    Unlike ``make_east_west_record``, this aggregates a season end to end with
    no date-level cumsum or shift: the season being summarized is already
    complete, so there is no "doesn't see its own result" concern to guard
    against. The caller (``src/etl/features/aggregator.py``) attaches each row
    to the *following* season via ``prev_season``
    (``src/etl/features/last_season_record.py``), so it reads as "what was
    last season's record" from that following season's games.

    Blended into the current season's record early on, before
    ``games_played_east_vs_west`` has accumulated enough of a sample — see
    ``src/ml/features/engineering.py::create_conference_features``.
    """
    east_west_record, suffix = _cross_conference_indicators(games, location=location)

    season_totals = (
        east_west_record.groupby("season")
        .agg(
            east_wins=("east_wins", "sum"),
            west_wins=("west_wins", "sum"),
            games_played_at_east=("east_games", "sum"),
            games_played_at_west=("west_games", "sum"),
            east_wins_at_east=("east_wins_at_east", "sum"),
            east_wins_at_west=("east_wins_at_west", "sum"),
        )
        .reset_index()
    )

    if location == "East":
        safe_gp = season_totals["games_played_at_east"].replace(0, float("nan"))
        season_totals[f"last_season_east_record{suffix}"] = (
            season_totals["east_wins"] / safe_gp
        ).fillna(0.0)
        return season_totals[["season", f"last_season_east_record{suffix}"]]

    if location == "West":
        safe_gp = season_totals["games_played_at_west"].replace(0, float("nan"))
        season_totals[f"last_season_west_record{suffix}"] = (
            season_totals["west_wins"] / safe_gp
        ).fillna(0.0)
        return season_totals[["season", f"last_season_west_record{suffix}"]]

    safe_gp_east = season_totals["games_played_at_east"].replace(0, float("nan"))
    safe_gp_west = season_totals["games_played_at_west"].replace(0, float("nan"))

    east_wr_at_east = season_totals["east_wins_at_east"] / safe_gp_east
    east_wr_at_west = season_totals["east_wins_at_west"] / safe_gp_west
    season_totals["last_season_east_record_adjusted"] = (
        pd.DataFrame({"at_east": east_wr_at_east, "at_west": east_wr_at_west})
        .mean(axis=1)
        .fillna(0.0)
    )

    west_wr_at_east = (
        season_totals["games_played_at_east"] - season_totals["east_wins_at_east"]
    ) / safe_gp_east
    west_wr_at_west = (
        season_totals["games_played_at_west"] - season_totals["east_wins_at_west"]
    ) / safe_gp_west
    season_totals["last_season_west_record_adjusted"] = (
        pd.DataFrame({"at_east": west_wr_at_east, "at_west": west_wr_at_west})
        .mean(axis=1)
        .fillna(0.0)
    )

    return season_totals[
        ["season", "last_season_east_record_adjusted", "last_season_west_record_adjusted"]
    ]
