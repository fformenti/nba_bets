"""Last season's win percentage record feature."""

import pandas as pd


def _prev_season(season: str) -> str:
    """Return the previous season string. "2024/25" -> "2023/24"."""
    start_year = int(season.split("/")[0])
    prev_start = start_year - 1
    prev_end = str(prev_start + 1)[-2:]
    return f"{prev_start}/{prev_end}"


def create_last_season_record(games: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each team's previous-season win percentage.

    For expansion teams (no prior season), fills with the minimum
    win percentage of all teams in that previous season.
    Returns NaN for teams in the first season of the dataset
    (handled by the ML pipeline's mean imputer).

    Parameters
    ----------
    games : pd.DataFrame
        Wide-format games with: season, hometeamId, awayteamId, winner.

    Returns
    -------
    pd.DataFrame
        Columns: teamId, season, last_season_record.
    """
    home = games[["season", "hometeamId", "winner"]].copy()
    home = home.rename(columns={"hometeamId": "teamId"})
    home["win_bool"] = (home["teamId"] == home["winner"]).astype(int)

    away = games[["season", "awayteamId", "winner"]].copy()
    away = away.rename(columns={"awayteamId": "teamId"})
    away["win_bool"] = (away["teamId"] == away["winner"]).astype(int)

    long = pd.concat(
        [home[["season", "teamId", "win_bool"]], away[["season", "teamId", "win_bool"]]],
        ignore_index=True,
    )

    season_record = (
        long.groupby(["teamId", "season"])["win_bool"]
        .mean()
        .round(4)
        .rename("win_pct")
        .reset_index()
    )

    season_record["prev_season"] = season_record["season"].map(_prev_season)

    # Look up each team's win_pct in their previous season
    prev_lookup = season_record[["teamId", "season", "win_pct"]].rename(
        columns={"season": "prev_season", "win_pct": "last_season_record"}
    )
    season_record = season_record.merge(prev_lookup, on=["teamId", "prev_season"], how="left")

    # Expansion teams: fill with that previous season's minimum
    prev_season_min = season_record.groupby("prev_season")["last_season_record"].transform("min")
    season_record["last_season_record"] = season_record["last_season_record"].fillna(prev_season_min)

    return season_record[["teamId", "season", "last_season_record"]].copy()


def create_last_season_home_record(games: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each team's previous-season win percentage at home.

    Parameters
    ----------
    games : pd.DataFrame
        Wide-format games with: season, hometeamId, winner.

    Returns
    -------
    pd.DataFrame
        Columns: teamId, season, last_season_record.
    """
    home = games[["season", "hometeamId", "winner"]].copy()
    home = home.rename(columns={"hometeamId": "teamId"})
    home["win_bool"] = (home["teamId"] == home["winner"]).astype(int)

    season_record = (
        home.groupby(["teamId", "season"])["win_bool"]
        .mean()
        .round(4)
        .rename("win_pct")
        .reset_index()
    )

    season_record["prev_season"] = season_record["season"].map(_prev_season)

    prev_lookup = season_record[["teamId", "season", "win_pct"]].rename(
        columns={"season": "prev_season", "win_pct": "last_season_record"}
    )
    season_record = season_record.merge(prev_lookup, on=["teamId", "prev_season"], how="left")

    prev_season_min = season_record.groupby("prev_season")["last_season_record"].transform("min")
    season_record["last_season_record"] = season_record["last_season_record"].fillna(prev_season_min)

    return season_record[["teamId", "season", "last_season_record"]].copy()


def create_last_season_away_record(games: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each team's previous-season win percentage on the road.

    Parameters
    ----------
    games : pd.DataFrame
        Wide-format games with: season, awayteamId, hometeamId, winner.

    Returns
    -------
    pd.DataFrame
        Columns: teamId, season, last_season_record.
    """
    away = games[["season", "awayteamId", "hometeamId", "winner"]].copy()
    away = away.rename(columns={"awayteamId": "teamId"})
    away["win_bool"] = (away["winner"] != away["hometeamId"]).astype(int)

    season_record = (
        away.groupby(["teamId", "season"])["win_bool"]
        .mean()
        .round(4)
        .rename("win_pct")
        .reset_index()
    )

    season_record["prev_season"] = season_record["season"].map(_prev_season)

    prev_lookup = season_record[["teamId", "season", "win_pct"]].rename(
        columns={"season": "prev_season", "win_pct": "last_season_record"}
    )
    season_record = season_record.merge(prev_lookup, on=["teamId", "prev_season"], how="left")

    prev_season_min = season_record.groupby("prev_season")["last_season_record"].transform("min")
    season_record["last_season_record"] = season_record["last_season_record"].fillna(prev_season_min)

    return season_record[["teamId", "season", "last_season_record"]].copy()
