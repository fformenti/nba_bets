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
from src.etl.utils.common import has_result


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

    played = has_result(games)

    home = games[base_cols + ["hometeamId", "awayteamId", "winner"]].copy()
    home["teamId"] = home["hometeamId"]
    home["opponentId"] = home["awayteamId"]
    home["win_bool"] = (home["hometeamId"] == home["winner"]).astype(int)
    home["has_result"] = played.to_numpy()

    away = games[base_cols + ["hometeamId", "awayteamId", "winner"]].copy()
    away["teamId"] = away["awayteamId"]
    away["opponentId"] = away["hometeamId"]
    away["win_bool"] = (away["awayteamId"] == away["winner"]).astype(int)
    away["has_result"] = played.to_numpy()

    keep = base_cols + ["teamId", "opponentId", "win_bool", "has_result"]
    return (
        pd.concat([home[keep], away[keep]], ignore_index=True)
        .sort_values(["teamId", "season", "gameDate"])
        .reset_index(drop=True)
    )


def _add_cumulative_win_pct(all_games: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative win percentage (including current game) per team-season.

    Only games that have been played count. A not-yet-played game carries
    ``winner = 0``, which reads as a loss for both teams, so counting it would
    push both their win percentages down for every later game that looks them
    up — including the other games on the same slate, which tip off at different
    times of the same day and therefore fall strictly after each other.

    Unplayed rows keep the win percentage as of the last played game, and are
    excluded from the lookup entirely (see ``_build_win_pct_lookup``).
    """
    groups = [all_games["teamId"], all_games["season"]]
    played = all_games["has_result"]

    all_games["cum_wins"] = (
        all_games["win_bool"].where(played, 0).groupby(groups).cumsum()
    )
    all_games["game_num"] = played.astype(int).groupby(groups).cumsum()
    all_games["cum_win_pct"] = all_games["cum_wins"] / all_games["game_num"].replace(
        0, np.nan
    )
    return all_games


def _build_win_pct_lookup(
    all_games: pd.DataFrame,
) -> dict[tuple, tuple[np.ndarray, np.ndarray]]:
    """Build dict mapping (teamId, season) → (sorted_dates, win_pcts).

    Played games only: an opponent's strength is what their results say it is,
    and a game with no result yet says nothing.
    """
    lookup: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    played_games = all_games[all_games["has_result"]]
    for (team_id, season), group in played_games.groupby(["teamId", "season"]):
        sorted_group = group.sort_values("gameDate")
        lookup[(team_id, season)] = (
            # Truncated to the day: an opponent's strength is read as of the
            # morning of game day, never mid-day. See _compute_sos_columns.
            sorted_group["gameDate"].values.astype("datetime64[D]"),
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
    """Compute SOS values for each lag window across all team-season groups.

    Opponent strength is evaluated as of the *start of the current game's day*,
    not its tip-off time. Two games on the same evening tip off hours apart, so
    a timestamp comparison let the later one read the earlier one's result — a
    result the prediction pipeline cannot have, because it predicts the whole
    slate before any of it is played. Training was consuming information serving
    can never see, which is the definition of train/serve skew.

    Day granularity is the convention ``compute_playoff_flags`` already uses for
    the same reason (standings as of date D, pre-game for teams playing on D).
    """
    sos_columns: dict[str, list[float]] = {f"sos_L{lag}": [] for lag in lags}
    prior_strength = prior_strength or {}

    for (_team_id, _season), group in all_games.groupby(
        ["teamId", "season"], sort=False
    ):
        sorted_group = group.sort_values("gameDate")
        dates = sorted_group["gameDate"].values.astype("datetime64[D]")
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
