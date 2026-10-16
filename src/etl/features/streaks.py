"""Win/loss streak feature calculations."""

import numpy as np
import pandas as pd


def _streak_from_results(all_games: pd.DataFrame) -> pd.DataFrame:
    """
    Shared core: consecutive win/loss streak per (teamId, season) group,
    entering each game, from a frame already carrying teamId and win_bool.

    Positive values indicate consecutive wins, negative values indicate
    consecutive losses. The first game of each group always has streak=0.
    Uses only the previous game's result to avoid data leakage.
    """
    all_games = all_games.sort_values(["teamId", "season", "gameDate", "gameId"])

    # Collect per-group results as index-aligned Series rather than one flat
    # list: a positional assignment would silently attach streaks to the wrong
    # rows if the sort keys or the groupby ever stopped agreeing.
    streak_parts: list[pd.Series] = []
    for _, group in all_games.groupby(["teamId", "season"], sort=False):
        group = group.sort_values(["gameDate", "gameId"])
        wins = group["win_bool"].values
        n = len(wins)
        streak = np.zeros(n, dtype=int)
        for i in range(1, n):
            if wins[i - 1] == 1:
                streak[i] = max(streak[i - 1], 0) + 1
            else:
                streak[i] = min(streak[i - 1], 0) - 1
        streak_parts.append(pd.Series(streak, index=group.index))

    all_games["streak"] = pd.concat(streak_parts)
    return all_games[["gameId", "season", "teamId", "gameDate", "streak"]]


def calculate_streak(games: pd.DataFrame) -> pd.DataFrame:
    """
    Compute consecutive win/loss streak for each team entering each game.

    Positive values indicate consecutive wins, negative values indicate
    consecutive losses. The first game of each season always has streak=0.

    Uses only the previous game's result to avoid data leakage.

    Parameters
    ----------
    games : pd.DataFrame
        Games DataFrame with columns hometeamId, awayteamId, winner,
        gameId, gameDate, season.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [gameId, season, teamId, gameDate, streak].
    """
    base_cols = ["gameId", "gameDate", "season"]

    home = games[base_cols + ["hometeamId", "winner"]].copy()
    home["teamId"] = home["hometeamId"]
    home["win_bool"] = (home["hometeamId"] == home["winner"]).astype(int)

    away = games[base_cols + ["awayteamId", "winner"]].copy()
    away["teamId"] = away["awayteamId"]
    away["win_bool"] = (away["awayteamId"] == away["winner"]).astype(int)

    all_games = pd.concat(
        [
            home[base_cols + ["teamId", "win_bool"]],
            away[base_cols + ["teamId", "win_bool"]],
        ],
        ignore_index=True,
    )
    return _streak_from_results(all_games)


def calculate_home_streak(games: pd.DataFrame) -> pd.DataFrame:
    """
    Consecutive win/loss streak using only each team's home games.

    Grouping by teamId (= hometeamId) naturally isolates each team's home
    appearances, the same way ``calculate_home_record``/``calculate_home_margin``
    do. See ``calculate_streak`` for the streak semantics.
    """
    base_cols = ["gameId", "gameDate", "season"]
    home = games[base_cols + ["hometeamId", "winner"]].copy()
    home["teamId"] = home["hometeamId"]
    home["win_bool"] = (home["hometeamId"] == home["winner"]).astype(int)
    return _streak_from_results(home[base_cols + ["teamId", "win_bool"]])


def calculate_away_streak(games: pd.DataFrame) -> pd.DataFrame:
    """
    Consecutive win/loss streak using only each team's away games.

    See ``calculate_home_streak``.
    """
    base_cols = ["gameId", "gameDate", "season"]
    away = games[base_cols + ["awayteamId", "winner"]].copy()
    away["teamId"] = away["awayteamId"]
    away["win_bool"] = (away["awayteamId"] == away["winner"]).astype(int)
    return _streak_from_results(away[base_cols + ["teamId", "win_bool"]])
