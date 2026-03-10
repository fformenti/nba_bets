from typing import Optional

import pandas as pd

from src.config.paths import (
    RAW_DISTANCES_PATH,
    REGULAR_SEASON_GAMES_PATH,
    TEAMS_CITIES_CONFERENCE_HISTORY_PROCESSED_PATH,
    TEAMS_DISTANCES_PATH,
)
from src.config.constants import TEAMS_CITIES_MAP


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
        "hometeamCity",
    ]
    home_games = games[home_cols]
    home_games.rename(columns={"hometeamId": "teamId"}, inplace=True)

    away_cols = [
        "gameId",
        "gameDate",
        "gameDateOnlyStr",
        "season",
        "awayteamId",
        "hometeamCity",
    ]
    away_games = games[away_cols]
    away_games.rename(columns={"awayteamId": "teamId"}, inplace=True)

    teams_game_dates = (
        pd.concat([home_games, away_games])
        .reset_index(drop=True)
        .rename(columns={"hometeamCity": "game_city"})
    )

    teams_info = pd.read_csv(TEAMS_CITIES_CONFERENCE_HISTORY_PROCESSED_PATH)

    teams_game_dates = teams_game_dates.merge(
        teams_info[["teamId", "teamCity", "season"]],
        on=["teamId", "season"],
        how="inner",
    )
    teams_game_dates["team_city_name"] = teams_game_dates["teamCity"].replace(
        TEAMS_CITIES_MAP
    )

    teams_game_dates["game_city_name"] = teams_game_dates["game_city"].replace(
        TEAMS_CITIES_MAP
    )

    return teams_game_dates


def conditional_backfill_group(group):
    group["current_city"] = group["current_city"].mask(
        group["current_city"].isna()
        & (group["next_game_city_name"].bfill() == group["team_city_name"]),
        group["next_game_city_name"].bfill(),
    )
    return group


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
    distances = pd.read_csv(RAW_DISTANCES_PATH)
    aux_distances = distances.rename(columns={"city1": "city2", "city2": "city1"})
    distances = pd.concat([distances, aux_distances]).reset_index(drop=True)

    teams_game_dates = make_teams_games_dates(games)
    distances_season = []
    seasons = teams_game_dates["season"].unique()
    for season in seasons:
        teams_game_dates_season = teams_game_dates[
            teams_game_dates["season"] == season
        ].sort_values(["teamId", "gameDate"])

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

        full_calendar_teams["team_city_name"] = full_calendar_teams["team_city_name"].ffill()
        full_calendar_teams["current_city"] = full_calendar_teams["game_city_name"]
        full_calendar_teams["next_game_city_name"] = full_calendar_teams.groupby(
            ["teamId"]
        )["game_city_name"].shift(-1)

        full_calendar_teams = full_calendar_teams.groupby(
            "teamId", group_keys=False
        ).apply(conditional_backfill_group)

        # to do: repensar se um basta eu fazer um backfill na tabela toda
        full_calendar_teams["current_city"] = full_calendar_teams.groupby("teamId")[
            "current_city"
        ].ffill()

        full_calendar_teams["team_city_name"] = full_calendar_teams.groupby("teamId")[
            "team_city_name"
        ].bfill()

        full_calendar_teams.drop(columns=["next_game_city_name"], inplace=True)
        full_calendar_teams[full_calendar_teams["teamCity"] == "New York"]

        full_calendar_teams["previous_city"] = full_calendar_teams.groupby(["teamId"])[
            "current_city"
        ].shift(1)

        # Fill Na for teams that didn't play the first game at home
        def if_na_select_column(group):
            group["previous_city"] = group["previous_city"].mask(
                group["previous_city"].isna(),
                group["team_city_name"],
            )
            return group

        full_calendar_teams = full_calendar_teams.groupby(
            "teamId", group_keys=False
        ).apply(if_na_select_column)
        # ---------------------

        # ---- Fill Na for teams that didn't play in the first day of the season
        full_calendar_teams["current_city"] = full_calendar_teams.groupby("teamId")[
            "current_city"
        ].bfill()

        full_calendar_teams = full_calendar_teams.merge(
            distances,
            left_on=["current_city", "previous_city"],
            right_on=["city1", "city2"],
            how="left",
        )
        # ---------------------

        # ---- Fill 0 for same city distances
        full_calendar_teams["distance"] = (
            full_calendar_teams["distance"].fillna(0).astype(int)
        )
        full_calendar_teams.drop(columns=["city1", "city2"], inplace=True)
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
    temas_distances = make_teams_distances_table_season(lags=[1, 7, 14])
    temas_distances.to_csv(TEAMS_DISTANCES_PATH, index=False)
