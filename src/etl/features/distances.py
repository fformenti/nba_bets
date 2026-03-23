from typing import Optional

import pandas as pd

from src.config.paths import (
    REGULAR_SEASON_GAMES_PATH,
    LOCATIONS_DISTANCES_PATH,
    TEAMS_DISTANCES_PATH,
)


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

        date_range = pd.date_range(start=start_date, end=end_date)
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

        full_calendar_teams["driving_distance"] = (
            full_calendar_teams["driving_distance"].fillna(0).astype(int)
        )

        full_calendar_teams.rename(
            columns={"driving_distance": "distance"}, inplace=True
        )
        # ---------------------

        # ---- Calculate distances lags L1 represents the current game distance
        distances_lags_cols = []
        for lag in lags:
            full_calendar_teams[f"distance_L{lag}"] = (
                full_calendar_teams.groupby(["teamId", "season"])["distance"]
                .transform(lambda x: x.rolling(window=int(lag), min_periods=1).mean())
                .round(0)
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


if __name__ == "__main__":
    teams_distances = make_teams_distances_table_season(lags=[1, 3, 7, 14])
    teams_distances.to_csv(TEAMS_DISTANCES_PATH, index=False)
