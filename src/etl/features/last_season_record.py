"""Last season's win percentage record feature."""

import pandas as pd

from src.etl.features.sos_adjusted_record import _apply_adjustment


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


def _season_sos(sos_df: pd.DataFrame) -> pd.DataFrame:
    """Extract end-of-season SOS per team per season using sos_L82."""
    last_rows = (
        sos_df.sort_values("gameDate")
        .groupby(["teamId", "season"])
        .last()
        .reset_index()[["teamId", "season", "sos_L82"]]
        .rename(columns={"sos_L82": "season_sos"})
    )
    return last_rows


def _apply_sos_to_season_record(
    season_record: pd.DataFrame,
    sos_df: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    """Merge season SOS into season_record and apply the multiplicative adjustment."""
    season_sos = _season_sos(sos_df)
    merged = season_record.merge(season_sos, on=["teamId", "season"], how="left")
    league_avg_sos = merged.groupby("season")["season_sos"].transform("mean")
    merged["win_pct"] = _apply_adjustment(
        merged["win_pct"], merged["season_sos"], league_avg_sos, alpha
    ).round(4)
    return merged.drop(columns=["season_sos"])


def create_adjusted_last_season_record(
    games: pd.DataFrame,
    sos_df: pd.DataFrame,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """
    Compute each team's previous-season win percentage adjusted for SOS.

    Applies the same multiplicative adjustment as ``sos_adj_record_L{lag}``:
        adjusted = win_pct * (team_sos / league_avg_sos) ^ alpha

    Season SOS is taken as each team's end-of-season ``sos_L82`` value.

    Parameters
    ----------
    games : pd.DataFrame
        Wide-format games with: season, hometeamId, awayteamId, winner.
    sos_df : pd.DataFrame
        Output of ``calculate_strength_of_schedule()``.  Must contain
        columns: teamId, season, gameDate, sos_L82.
    alpha : float
        Scaling exponent (same as ``sos_adj_alpha`` in features.yaml).

    Returns
    -------
    pd.DataFrame
        Columns: teamId, season, adjusted_last_season_record.
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

    season_record = _apply_sos_to_season_record(season_record, sos_df, alpha)

    season_record["prev_season"] = season_record["season"].map(_prev_season)

    prev_lookup = season_record[["teamId", "season", "win_pct"]].rename(
        columns={"season": "prev_season", "win_pct": "adjusted_last_season_record"}
    )
    season_record = season_record.merge(prev_lookup, on=["teamId", "prev_season"], how="left")

    prev_season_min = season_record.groupby("prev_season")["adjusted_last_season_record"].transform("min")
    season_record["adjusted_last_season_record"] = season_record["adjusted_last_season_record"].fillna(prev_season_min)

    return season_record[["teamId", "season", "adjusted_last_season_record"]].copy()


def create_adjusted_last_season_home_record(
    games: pd.DataFrame,
    sos_df: pd.DataFrame,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """
    Compute each team's previous-season home win percentage adjusted for SOS.

    Returns
    -------
    pd.DataFrame
        Columns: teamId, season, adjusted_last_season_record.
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

    season_record = _apply_sos_to_season_record(season_record, sos_df, alpha)

    season_record["prev_season"] = season_record["season"].map(_prev_season)

    prev_lookup = season_record[["teamId", "season", "win_pct"]].rename(
        columns={"season": "prev_season", "win_pct": "adjusted_last_season_record"}
    )
    season_record = season_record.merge(prev_lookup, on=["teamId", "prev_season"], how="left")

    prev_season_min = season_record.groupby("prev_season")["adjusted_last_season_record"].transform("min")
    season_record["adjusted_last_season_record"] = season_record["adjusted_last_season_record"].fillna(prev_season_min)

    return season_record[["teamId", "season", "adjusted_last_season_record"]].copy()


def create_adjusted_last_season_away_record(
    games: pd.DataFrame,
    sos_df: pd.DataFrame,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """
    Compute each team's previous-season road win percentage adjusted for SOS.

    Returns
    -------
    pd.DataFrame
        Columns: teamId, season, adjusted_last_season_record.
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

    season_record = _apply_sos_to_season_record(season_record, sos_df, alpha)

    season_record["prev_season"] = season_record["season"].map(_prev_season)

    prev_lookup = season_record[["teamId", "season", "win_pct"]].rename(
        columns={"season": "prev_season", "win_pct": "adjusted_last_season_record"}
    )
    season_record = season_record.merge(prev_lookup, on=["teamId", "prev_season"], how="left")

    prev_season_min = season_record.groupby("prev_season")["adjusted_last_season_record"].transform("min")
    season_record["adjusted_last_season_record"] = season_record["adjusted_last_season_record"].fillna(prev_season_min)

    return season_record[["teamId", "season", "adjusted_last_season_record"]].copy()


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
