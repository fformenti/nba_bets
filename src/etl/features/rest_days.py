"""Rest days between games feature calculations."""

import numpy as np
import pandas as pd
from pandas import DataFrame
from numpy import nan

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Consecutive days at home / on the road above this are treated as a league
# stoppage (e.g. the 2020 COVID suspension) rather than a real homestand or trip.
COVID_BREAK_THRESHOLD = 30


def is_home(home_game, away_game):
    """Determine if team is playing at home."""
    if home_game == 1:
        return 1
    elif away_game == 1:
        return 0
    else:
        return nan


def is_away(home_game, away_game):
    """Determine if team is playing away."""
    if home_game == 1:
        return 0
    elif away_game == 1:
        return 1
    else:
        return nan


def make_rested_days_table(games):
    """
    Calculate rest days between games for all teams.

    Parameters
    ----------
    games : pd.DataFrame
        Games DataFrame with gameDate, season, hometeamId, awayteamId columns

    Returns
    -------
    pd.DataFrame
        DataFrame with rested_days, days_at_home, days_on_road columns
    """
    rested_days_season = []
    seasons = games["season"].unique()
    for season in seasons:
        games_season = games.loc[games["season"] == season].copy()
        season_start = games_season["gameDate"].min()
        season_end = games_season["gameDate"].max()
        # Union of both sides: a team that never hosts in a season would
        # otherwise get no rows in the calendar grid, and so no rest features.
        season_teams_ids = pd.unique(
            pd.concat([games_season["hometeamId"], games_season["awayteamId"]])
        )

        # Days in Between Games
        rested_days_season.append(
            make_rested_days_table_season(
                games_season, season_start, season_end, season_teams_ids
            )
        )

    return pd.concat(rested_days_season, ignore_index=True)


def make_rested_days_table_season(
    games_filtered, start_date, end_date, teams_season
) -> DataFrame:
    """
    Calculate rest days for a single season.

    Parameters
    ----------
    games_filtered : pd.DataFrame
        Games for a single season
    start_date : pd.Timestamp
        Season start date
    end_date : pd.Timestamp
        Season end date
    teams_season : array-like
        Team IDs for the season

    Returns
    -------
    pd.DataFrame
        Rest days DataFrame
    """
    # Normalize to midnight: pd.date_range carries the start's time-of-day, so a
    # finale tipping off earlier in the day than the opener would fall past `end`
    # and drop the season's last day entirely.
    date_range = pd.date_range(
        start=start_date.normalize(), end=end_date.normalize()
    )
    rested_days = pd.MultiIndex.from_product(
        [date_range, teams_season], names=["gameDate", "teamId"]
    ).to_frame(index=False)
    rested_days["gameDateOnlyStr"] = rested_days["gameDate"].dt.strftime("%Y-%m-%d")

    # drop_duplicates is required: these merges only ask "did this team play at
    # home/away on this date?", so a team with two games on one calendar day
    # would otherwise duplicate its grid row and every downstream join.
    rested_days = rested_days.merge(
        games_filtered[["gameDateOnlyStr", "hometeamId"]].drop_duplicates(),
        left_on=["gameDateOnlyStr", "teamId"],
        right_on=["gameDateOnlyStr", "hometeamId"],
        how="left",
        indicator="home_game",
    ).drop("hometeamId", axis=1)

    rested_days = rested_days.merge(
        games_filtered[["gameDateOnlyStr", "awayteamId"]].drop_duplicates(),
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

    # Back-to-back: how many consecutive days immediately before this one the
    # team also played. 0 = rested yesterday, 1 = back-to-back, 2 = three games
    # in three days, and so on.
    #
    # `rest` is still unshifted here, so (1 - rest) is "played today". Runs of
    # consecutive playing days are broken by any rest day; the run counter minus
    # one gives the number of preceding games in the current run.
    rested_days["played"] = 1 - rested_days["rest"]
    play_run_id = (rested_days["played"] == 0).cumsum()
    rested_days["back_to_back"] = (
        rested_days.groupby(["teamId", play_run_id])["played"].cumsum() - 1
    ).clip(lower=0)

    rested_days["rest"] = (
        rested_days.groupby("teamId")["rest"].shift(1).fillna(1).astype(int)
    )

    rested_days["rested_days"] = rested_days.groupby(
        ["teamId", (rested_days["rest"] == 0).cumsum()]
    )["rest"].transform("cumsum")

    # At Home Count
    #
    # Vectorised: this frame is every team × every calendar day of every season,
    # so a row-wise apply here crossed into Python about a million times and
    # dominated the whole feature build. np.select reproduces is_home() exactly,
    # including the NaN for a day with no game (which the ffill below fills).
    home_game = rested_days["home_game"]
    away_game = rested_days["away_game"]
    played_home = home_game == 1
    played_away = away_game == 1

    rested_days["at_home_indicator"] = np.select(
        [played_home, played_away], [1.0, 0.0], default=nan
    )
    rested_days["at_home_indicator"] = (
        rested_days.groupby("teamId")["at_home_indicator"].ffill().fillna(1).astype(int)
    )
    rested_days["days_at_home"] = rested_days.groupby(
        ["teamId", (rested_days["at_home_indicator"] == 0).cumsum()]
    )["at_home_indicator"].transform("cumsum")

    # On the Road Count (same vectorisation as above)
    rested_days["at_road_indicator"] = np.select(
        [played_home, played_away], [0.0, 1.0], default=nan
    )
    rested_days["at_road_indicator"] = (
        rested_days.groupby("teamId")["at_road_indicator"].ffill().fillna(0).astype(int)
    )
    rested_days["days_on_road"] = rested_days.groupby(
        ["teamId", (rested_days["at_road_indicator"] == 0).cumsum()]
    )["at_road_indicator"].transform("cumsum")

    # Fix league-stoppage outliers (COVID). A team idle at home for months would
    # otherwise dwarf every real homestand.
    #
    # days_at_home is capped at this season's largest non-outlier value, so the
    # ordering stays monotone and on the same scale as the rest of the season.
    # days_on_road resets to 1: teams went home during the break, so the first
    # game back is day one of a new road trip.
    at_home_outliers = rested_days["days_at_home"] > COVID_BREAK_THRESHOLD
    on_road_outliers = rested_days["days_on_road"] > COVID_BREAK_THRESHOLD

    if at_home_outliers.any() or on_road_outliers.any():
        normal = rested_days.loc[~at_home_outliers, "days_at_home"]
        cap = int(normal.max()) if len(normal) else COVID_BREAK_THRESHOLD
        logger.warning(
            "Rest days: clamping league-stoppage outliers "
            "(days_at_home>%d: %d rows -> %d, days_on_road>%d: %d rows -> 1)",
            COVID_BREAK_THRESHOLD,
            int(at_home_outliers.sum()),
            cap,
            COVID_BREAK_THRESHOLD,
            int(on_road_outliers.sum()),
        )
        rested_days.loc[at_home_outliers, "days_at_home"] = cap
        rested_days.loc[on_road_outliers, "days_on_road"] = 1

    rested_days = rested_days.drop(
        columns=["gameDate", "home_game", "away_game", "rest", "played"]
    )

    return rested_days
