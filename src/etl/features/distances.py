from typing import Optional

import pandas as pd

from src.config.paths import (
    REGULAR_SEASON_GAMES_PATH,
    LOCATIONS_DISTANCES_PATH,
    TEAMS_LOCATIONS_REFERENCE_PATH,
)
from src.etl.utils.common import require_reference_file
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def make_teams_games_dates(games: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Build teams-game-dates table for distance calculations.

    Parameters
    ----------
    games : pd.DataFrame, optional
        Games DataFrame with gameId, gameDate, gameDateOnlyStr, season,
        hometeamId, awayteamId, hometeamCity. If None, reads from
        REGULAR_SEASON_GAMES_PATH. Use this when predicting upcoming games
        to include them in the distances table.
    """
    if games is None:
        games = pd.read_csv(REGULAR_SEASON_GAMES_PATH)
    else:
        games = games.copy()
        if "gameDate" in games.columns and not pd.api.types.is_datetime64_any_dtype(
            games["gameDate"]
        ):
            games["gameDate"] = pd.to_datetime(games["gameDate"], errors="coerce")
        if "gameDateOnlyStr" not in games.columns and "gameDate" in games.columns:
            games["gameDateOnlyStr"] = games["gameDate"].dt.strftime("%Y-%m-%d")
        if "hometeamCity" not in games.columns and "homeTeamCity" in games.columns:
            games["hometeamCity"] = games["homeTeamCity"]

    home_cols = [
        "gameId",
        "gameDate",
        "gameDateOnlyStr",
        "season",
        "hometeamId",
        "hometeamPrename",
        "hometeamName",
        "hometeamLocation",
        "gameLocation",
    ]
    home_games = games[home_cols].copy().rename(
        columns={
            "hometeamId": "teamId",
            "hometeamPrename": "teamPrename",
            "hometeamName": "teamName",
            "hometeamLocation": "teamLocation",
        }
    )

    away_cols = [
        "gameId",
        "gameDate",
        "gameDateOnlyStr",
        "season",
        "awayteamId",
        "awayteamPrename",
        "awayteamName",
        "awayteamLocation",
        "gameLocation",
    ]
    away_games = games[away_cols].copy().rename(
        columns={
            "awayteamId": "teamId",
            "awayteamPrename": "teamPrename",
            "awayteamName": "teamName",
            "awayteamLocation": "teamLocation",
        }
    )

    return pd.concat([home_games, away_games]).reset_index(drop=True)



def make_teams_distances_table_season(
    lags: list, games: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Build team distances table with optional games for prediction.

    Parameters
    ----------
    lags : list
        List of lags for rolling distance averages (e.g. [1, 3, 7, 14]).
    games : pd.DataFrame, optional
        Games to include. When provided (e.g. historical + upcoming for
        prediction), distances are computed for all games including
        upcoming ones. When None, uses REGULAR_SEASON_GAMES_PATH.
    """
    require_reference_file(LOCATIONS_DISTANCES_PATH, "build-distances-table")
    distances = pd.read_csv(LOCATIONS_DISTANCES_PATH)
    aux_distances = distances.rename(columns={"from": "to", "to": "from"})
    distances = pd.concat([distances, aux_distances]).reset_index(drop=True)

    teams_game_dates = make_teams_games_dates(games)
    distances_season = []
    seasons = teams_game_dates["season"].unique()
    for season in seasons:
        teams_game_dates_season = teams_game_dates[
            teams_game_dates["season"] == season
        ].sort_values(["teamId", "gameDate"])

        teams_game_dates_season = teams_game_dates_season.drop_duplicates(
            subset=["teamId", "gameDateOnlyStr"], keep="first"
        )

        # games_season = games.loc[games["season"] == season].copy()
        season_start = teams_game_dates_season["gameDate"].min()
        season_end = teams_game_dates_season["gameDate"].max()

        start_date = season_start
        end_date = season_end
        teams_season = teams_game_dates_season["teamId"].unique()

        # Normalize to midnight: pd.date_range carries the start's time-of-day,
        # so a finale tipping off earlier in the day than the opener would fall
        # past `end` and drop the season's last day entirely.
        date_range = pd.date_range(
            start=start_date.normalize(), end=end_date.normalize()
        )
        full_calendar_teams = pd.MultiIndex.from_product(
            [date_range, teams_season], names=["gameDate", "teamId"]
        ).to_frame(index=False)
        full_calendar_teams["gameDateOnlyStr"] = full_calendar_teams[
            "gameDate"
        ].dt.strftime("%Y-%m-%d")

        full_calendar_teams.drop(columns=["gameDate"], inplace=True)
        full_calendar_teams = full_calendar_teams.merge(
            teams_game_dates_season, on=["teamId", "gameDateOnlyStr"], how="left"
        )
        full_calendar_teams = full_calendar_teams.sort_values(
            ["teamId", "gameDateOnlyStr"]
        ).reset_index(drop=True)

        with pd.option_context("future.no_silent_downcasting", True):
            full_calendar_teams["teamLocation"] = full_calendar_teams[
                "teamLocation"
            ].ffill().infer_objects(copy=False)
            full_calendar_teams["current_location"] = full_calendar_teams["gameLocation"]
            full_calendar_teams["next_gameLocation"] = full_calendar_teams.groupby(
                ["teamId"]
            )["gameLocation"].shift(-1)

            _next_filled = full_calendar_teams.groupby("teamId")["next_gameLocation"].transform(
                lambda x: x.bfill().infer_objects(copy=False)
            )
        full_calendar_teams["current_location"] = full_calendar_teams["current_location"].mask(
            full_calendar_teams["current_location"].isna()
            & (_next_filled == full_calendar_teams["teamLocation"]),
            _next_filled,
        )

        # to do: repensar se um basta eu fazer um backfill na tabela toda
        full_calendar_teams["current_location"] = full_calendar_teams.groupby("teamId")[
            "current_location"
        ].ffill()

        full_calendar_teams["teamLocation"] = full_calendar_teams.groupby("teamId")[
            "teamLocation"
        ].bfill()

        full_calendar_teams.drop(columns=["next_gameLocation"], inplace=True)

        full_calendar_teams["previous_location"] = full_calendar_teams.groupby(
            ["teamId"]
        )["current_location"].shift(1)

        # Fill Na for teams that didn't play the first game at home
        full_calendar_teams["previous_location"] = full_calendar_teams[
            "previous_location"
        ].mask(
            full_calendar_teams["previous_location"].isna(),
            full_calendar_teams["teamLocation"],
        )
        # ---------------------

        # ---- Fill Na for teams that didn't play in the first day of the season
        full_calendar_teams["current_location"] = full_calendar_teams.groupby("teamId")[
            "current_location"
        ].bfill()

        # ---- Merge distances table
        full_calendar_teams = full_calendar_teams.merge(
            distances[["from", "to", "driving_distance"]],
            left_on=["current_location", "previous_location"],
            right_on=["from", "to"],
            how="left",
        ).drop(columns=["from", "to"])

        # An unmatched pair means one of two very different things. The distance
        # table holds *distinct* city pairs only (itertools.combinations), so a
        # team that stayed put has no row by construction and genuinely travelled
        # zero. A pair of different cities with no row is a gap in the reference
        # table, and filling it with 0 would read as "no travel" — the same value
        # as a rest day at home. Only the first case is filled silently.
        # A null location is a third case again — a team-season the locations
        # lookup does not cover — and cannot be judged either way, so it is
        # counted and left at zero rather than failing the build.
        known = (
            full_calendar_teams["current_location"].notna()
            & full_calendar_teams["previous_location"].notna()
        )
        unknown_location = full_calendar_teams["driving_distance"].isna() & ~known
        if unknown_location.any():
            logger.warning(
                "Season %s: %d team-day(s) have no known location; distance recorded "
                "as 0. Check the team-season coverage of %s.",
                season,
                int(unknown_location.sum()),
                TEAMS_LOCATIONS_REFERENCE_PATH.name,
            )

        same_city = (
            full_calendar_teams["current_location"]
            == full_calendar_teams["previous_location"]
        )
        unmatched = full_calendar_teams["driving_distance"].isna() & ~same_city & known
        if unmatched.any():
            missing_pairs = (
                full_calendar_teams.loc[unmatched, ["previous_location", "current_location"]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )
            raise ValueError(
                f"{int(unmatched.sum())} team-day(s) in season {season} travelled "
                f"between cities missing from {LOCATIONS_DISTANCES_PATH.name}: "
                f"{sorted(missing_pairs)}. Add them rather than letting the trip "
                f"read as zero miles."
            )

        full_calendar_teams["driving_distance"] = (
            full_calendar_teams["driving_distance"].fillna(0).astype(int)
        )

        full_calendar_teams.rename(
            columns={"driving_distance": "distance"}, inplace=True
        )
        # ---------------------

        # ---- Calculate distances lags L1 represents the current game distance
        #
        # Deliberately *not* shifted, unlike every other rolling feature in the
        # ETL. Travel into a game is fixed by the schedule and fully known before
        # tip-off, so including the current row is information the model would
        # genuinely have at prediction time — not lookahead. This is the one
        # exception to the shift(1) rule; keep it documented as such.
        distances_lags_cols = []
        for lag in lags:
            full_calendar_teams[f"distance_L{lag}"] = (
                full_calendar_teams.groupby(["teamId", "season"])["distance"]
                .transform(lambda x: x.rolling(window=int(lag), min_periods=1).mean())
            )
            distances_lags_cols.append(f"distance_L{lag}")

        keep_cols = [
            "gameId",
            "teamId",
            "season",
            "gameDateOnlyStr",
            # "distance",
        ] + distances_lags_cols
        distances_season.append(full_calendar_teams[keep_cols])

    return pd.concat(distances_season, ignore_index=True)
