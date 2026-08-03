"""Strength of Schedule (SOS) feature calculation.

For each team-game row, computes the rolling average win percentage of
the team's previous N opponents, where each opponent's win percentage
is evaluated as of the current game's date.
"""

import numpy as np
import pandas as pd

from src.etl.features.last_season_record import (
    NEW_FRANCHISE_STRENGTH,
    build_prior_season_strength,
)


def calculate_strength_of_schedule(
    games: pd.DataFrame,
    lags: list[int],
    min_opponents: int = 3,
) -> pd.DataFrame:
    """
    Calculate rolling Strength of Schedule for each team per game.

    For each game, looks back at the team's last N opponents (within the
    season) and averages their overall season win percentage as of the
    current game's date.

    Parameters
    ----------
    games : pd.DataFrame
        Game-level DataFrame (one row per game) with columns:
        hometeamId, awayteamId, winner, gameId, gameDate, season.
    lags : list[int]
        Rolling window sizes. Each produces a column ``sos_L{lag}``.
    min_opponents : int, default 3
        Minimum number of previous opponents required to produce a value.
        Returns NaN if fewer.

    Notes
    -----
    An opponent who has not played yet this season has no current-season win
    percentage; their prior-season record is used instead (see
    ``build_prior_season_strength``).  Without this, early-season SOS
    collapses to 0.0 or NaN for whole slates of games.

    Returns
    -------
    pd.DataFrame
        Team-level DataFrame with columns:
        gameId, season, teamId, gameDate, sos_L{lag1}, sos_L{lag2}, ...
    """
    if not lags:
        all_games = _build_team_game_table(games)
        return all_games[["gameId", "season", "teamId", "gameDate"]].copy()

    all_games = _build_team_game_table(games)
    all_games = _add_cumulative_win_pct(all_games)
    lookup = _build_win_pct_lookup(all_games)
    prior_strength = _build_prior_strength_lookup(games)

    sos_columns = _compute_sos_columns(
        all_games, lookup, lags, min_opponents, prior_strength
    )
    for col_name, values in sos_columns.items():
        all_games[col_name] = values

    return_cols = ["gameId", "season", "teamId", "gameDate"] + [
        f"sos_L{lag}" for lag in lags
    ]
    return all_games[return_cols].copy()


def _build_team_game_table(games: pd.DataFrame) -> pd.DataFrame:
    """Split game-level rows into team-level rows (one per team per game)."""
    base_cols = ["gameId", "gameDate", "season"]

    home = games[base_cols + ["hometeamId", "awayteamId", "winner"]].copy()
    home["teamId"] = home["hometeamId"]
    home["opponentId"] = home["awayteamId"]
    home["win_bool"] = (home["hometeamId"] == home["winner"]).astype(int)

    away = games[base_cols + ["hometeamId", "awayteamId", "winner"]].copy()
    away["teamId"] = away["awayteamId"]
    away["opponentId"] = away["hometeamId"]
    away["win_bool"] = (away["awayteamId"] == away["winner"]).astype(int)

    keep = base_cols + ["teamId", "opponentId", "win_bool"]
    return (
        pd.concat([home[keep], away[keep]], ignore_index=True)
        .sort_values(["teamId", "season", "gameDate"])
        .reset_index(drop=True)
    )


def _add_cumulative_win_pct(all_games: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative win percentage (including current game) per team-season."""
    all_games["cum_wins"] = all_games.groupby(["teamId", "season"])[
        "win_bool"
    ].cumsum()
    all_games["game_num"] = (
        all_games.groupby(["teamId", "season"]).cumcount() + 1
    )
    all_games["cum_win_pct"] = all_games["cum_wins"] / all_games["game_num"]
    return all_games


def _build_win_pct_lookup(
    all_games: pd.DataFrame,
) -> dict[tuple, tuple[np.ndarray, np.ndarray]]:
    """Build dict mapping (teamId, season) → (sorted_dates, win_pcts)."""
    lookup: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    for (team_id, season), group in all_games.groupby(["teamId", "season"]):
        sorted_group = group.sort_values("gameDate")
        lookup[(team_id, season)] = (
            sorted_group["gameDate"].values,
            sorted_group["cum_win_pct"].values,
        )
    return lookup


def _build_prior_strength_lookup(games: pd.DataFrame) -> dict[tuple, float]:
    """Build dict mapping (teamId, season) → prior-season win percentage."""
    prior = build_prior_season_strength(games)
    return {
        (team_id, season): strength
        for team_id, season, strength in prior.itertuples(index=False)
    }


def _compute_sos_columns(
    all_games: pd.DataFrame,
    lookup: dict[tuple, tuple[np.ndarray, np.ndarray]],
    lags: list[int],
    min_opponents: int,
    prior_strength: dict[tuple, float] | None = None,
) -> dict[str, list[float]]:
    """Compute SOS values for each lag window across all team-season groups."""
    sos_columns: dict[str, list[float]] = {f"sos_L{lag}": [] for lag in lags}
    prior_strength = prior_strength or {}

    for (_team_id, _season), group in all_games.groupby(
        ["teamId", "season"], sort=False
    ):
        sorted_group = group.sort_values("gameDate")
        dates = sorted_group["gameDate"].values
        opponents = sorted_group["opponentId"].values
        n = len(sorted_group)

        for lag in lags:
            col_values = np.full(n, np.nan)
            for i in range(n):
                current_date = dates[i]
                start_idx = max(0, i - lag)
                prev_opponents = opponents[start_idx:i]

                if len(prev_opponents) < min_opponents:
                    continue

                opp_win_pcts = []
                for opp_id in prev_opponents:
                    key = (opp_id, _season)
                    idx = -1
                    if key in lookup:
                        opp_dates, opp_pcts = lookup[key]
                        idx = np.searchsorted(opp_dates, current_date, side="left") - 1
                    if idx >= 0:
                        opp_win_pcts.append(opp_pcts[idx])
                    else:
                        # Opponent has no completed games this season yet;
                        # use their prior-season strength instead of dropping them.
                        opp_win_pcts.append(
                            prior_strength.get(key, NEW_FRANCHISE_STRENGTH)
                        )

                if len(opp_win_pcts) >= min_opponents:
                    col_values[i] = np.mean(opp_win_pcts)

            sos_columns[f"sos_L{lag}"].extend(col_values)

    return sos_columns
