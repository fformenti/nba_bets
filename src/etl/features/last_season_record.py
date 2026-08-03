"""Last season's win percentage record feature.

Six public builders share the same shape: aggregate a season win percentage
over some scope (all games / home only / road only), optionally adjust it for
strength of schedule, then attach each team's *previous* season value. They
differ only in scope, whether the SOS adjustment is applied, and the output
column name — see ``_team_season_win_pct`` and ``_prior_season_lookup``.
"""

import pandas as pd

from src.etl.features.sos_adjusted_record import _apply_adjustment

# Assumed strength for a franchise with no prior season in the dataset
# (expansion teams such as the 1995/96 Raptors, and every team in the
# first season on record). Used as the terminal fallback for opponent
# quality in SOS and GDS.
NEW_FRANCHISE_STRENGTH = 0.200


def _prev_season(season: str) -> str:
    """Return the previous season string. "2024/25" -> "2023/24"."""
    start_year = int(season.split("/")[0])
    prev_start = start_year - 1
    prev_end = str(prev_start + 1)[-2:]
    return f"{prev_start}/{prev_end}"


def _team_season_win_pct(games: pd.DataFrame, scope: str = "all") -> pd.DataFrame:
    """
    Win percentage per team per season.

    Parameters
    ----------
    games : pd.DataFrame
        Wide-format games with: season, hometeamId, awayteamId, winner.
    scope : {"all", "home", "away"}
        Which games to count: every game, home games only, or road games only.

    Returns
    -------
    pd.DataFrame
        Columns: teamId, season, win_pct.
    """
    if scope not in ("all", "home", "away"):
        raise ValueError(f"scope must be 'all', 'home' or 'away', got {scope!r}")

    frames = []

    # Both sides test equality against the team itself. Testing the away side as
    # `winner != hometeamId` would be equivalent for played games, but would turn
    # an unknown winner (NA, or the 0 placeholder used for upcoming games) into a
    # fabricated road win while the home side recorded a loss.
    if scope in ("all", "home"):
        home = games[["season", "hometeamId", "winner"]].rename(
            columns={"hometeamId": "teamId"}
        )
        frames.append(home.assign(win_bool=(home["teamId"] == home["winner"]).astype(int)))

    if scope in ("all", "away"):
        away = games[["season", "awayteamId", "winner"]].rename(
            columns={"awayteamId": "teamId"}
        )
        frames.append(away.assign(win_bool=(away["teamId"] == away["winner"]).astype(int)))

    long = pd.concat(
        [frame[["season", "teamId", "win_bool"]] for frame in frames],
        ignore_index=True,
    )

    return (
        long.groupby(["teamId", "season"])["win_bool"]
        .mean()
        .rename("win_pct")
        .reset_index()
    )


def _prior_season_lookup(season_record: pd.DataFrame, out_col: str) -> pd.DataFrame:
    """
    Attach each team's *previous* season ``win_pct`` as *out_col*.

    Expansion teams (no prior season on record) fall back to the minimum
    prior-season value among teams playing in the same season. Teams in the
    first season of the dataset stay NaN, since no prior season exists at all;
    the ML pipeline's imputer handles those.

    Parameters
    ----------
    season_record : pd.DataFrame
        Columns: teamId, season, win_pct.
    out_col : str
        Name for the resulting previous-season column.

    Returns
    -------
    pd.DataFrame
        Columns: teamId, season, {out_col}.
    """
    season_record = season_record.copy()
    season_record["prev_season"] = season_record["season"].map(_prev_season)

    prev_lookup = season_record[["teamId", "season", "win_pct"]].rename(
        columns={"season": "prev_season", "win_pct": out_col}
    )
    season_record = season_record.merge(
        prev_lookup, on=["teamId", "prev_season"], how="left"
    )

    prev_season_min = season_record.groupby("prev_season")[out_col].transform("min")
    season_record[out_col] = season_record[out_col].fillna(prev_season_min)

    return season_record[["teamId", "season", out_col]].copy()


def build_prior_season_strength(
    games: pd.DataFrame,
    default: float = NEW_FRANCHISE_STRENGTH,
) -> pd.DataFrame:
    """
    Prior-season win percentage per team-season, used as a strength prior.

    Unlike ``create_last_season_record()`` (which back-fills expansion teams
    with the previous season's league minimum), teams with no prior season
    get a fixed *default*.  This is the fallback for opponent quality at the
    start of a season, when no current-season games have been played yet.

    Parameters
    ----------
    games : pd.DataFrame
        Wide-format games with: season, hometeamId, awayteamId, winner.
    default : float
        Strength assigned to franchises with no prior season on record.

    Returns
    -------
    pd.DataFrame
        Columns: teamId, season, prior_season_strength.
    """
    season_record = _team_season_win_pct(games)
    season_record["prev_season"] = season_record["season"].map(_prev_season)

    prev_lookup = season_record[["teamId", "season", "win_pct"]].rename(
        columns={"season": "prev_season", "win_pct": "prior_season_strength"}
    )
    season_record = season_record.merge(
        prev_lookup, on=["teamId", "prev_season"], how="left"
    )
    season_record["prior_season_strength"] = season_record[
        "prior_season_strength"
    ].fillna(default)

    return season_record[["teamId", "season", "prior_season_strength"]].copy()


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
    )
    return merged.drop(columns=["season_sos"])


# ── Plain (unadjusted) previous-season records ───────────────────────────────


def create_last_season_record(games: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each team's previous-season win percentage.

    For expansion teams (no prior season), fills with the minimum
    previous-season win percentage among teams playing that season.
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
    return _prior_season_lookup(
        _team_season_win_pct(games, scope="all"), "last_season_record"
    )


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
    return _prior_season_lookup(
        _team_season_win_pct(games, scope="home"), "last_season_record"
    )


def create_last_season_away_record(games: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each team's previous-season win percentage on the road.

    Parameters
    ----------
    games : pd.DataFrame
        Wide-format games with: season, awayteamId, winner.

    Returns
    -------
    pd.DataFrame
        Columns: teamId, season, last_season_record.
    """
    return _prior_season_lookup(
        _team_season_win_pct(games, scope="away"), "last_season_record"
    )


# ── SOS-adjusted previous-season records ─────────────────────────────────────


def _create_adjusted_last_season_record(
    games: pd.DataFrame,
    sos_df: pd.DataFrame,
    alpha: float,
    scope: str,
) -> pd.DataFrame:
    """Shared body for the three SOS-adjusted previous-season builders."""
    season_record = _team_season_win_pct(games, scope=scope)
    season_record = _apply_sos_to_season_record(season_record, sos_df, alpha)
    return _prior_season_lookup(season_record, "adjusted_last_season_record")


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
    return _create_adjusted_last_season_record(games, sos_df, alpha, scope="all")


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
    return _create_adjusted_last_season_record(games, sos_df, alpha, scope="home")


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
    return _create_adjusted_last_season_record(games, sos_df, alpha, scope="away")
