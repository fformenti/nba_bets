"""Win/loss streak feature calculations."""

import numpy as np
import pandas as pd


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
    ).sort_values(["teamId", "season", "gameDate"])

    streak_values: list[int] = []
    for _, group in all_games.groupby(["teamId", "season"], sort=False):
        wins = group.sort_values("gameDate")["win_bool"].values
        n = len(wins)
        streak = np.zeros(n, dtype=int)
        for i in range(1, n):
            if wins[i - 1] == 1:
                streak[i] = max(streak[i - 1], 0) + 1
            else:
                streak[i] = min(streak[i - 1], 0) - 1
        streak_values.extend(streak)

    all_games["streak"] = streak_values
    return all_games[["gameId", "season", "teamId", "gameDate", "streak"]]
