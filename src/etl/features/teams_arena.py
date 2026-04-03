"""Build per-team home arena lookup and derive the neutral_court game flag."""

import ast

import pandas as pd


# Any arena where the team hosted at least this fraction of home games is
# considered a "home arena". A value of 0.20 accommodates split seasons
# (e.g. New Orleans Hornets 2005 post-Katrina).
_HOME_ARENA_THRESHOLD = 0.20


def build_teams_arena(games: pd.DataFrame) -> pd.DataFrame:
    """Compute home arena(s) for each (team, season) pair.

    Parameters
    ----------
    games : pd.DataFrame
        Regular-season games with columns ``hometeamId``, ``season``,
        ``arenaId``. Rows with a null ``arenaId`` are ignored.

    Returns
    -------
    pd.DataFrame
        Columns: ``team_id`` (int), ``season`` (str),
        ``home_arena_ids`` (list[int]).
        One row per (team, season) that has at least one game with a
        known arenaId.
    """
    arena_games = games[games["arenaId"].notna()][
        ["hometeamId", "season", "arenaId"]
    ].copy()
    arena_games["arenaId"] = arena_games["arenaId"].astype(int)

    counts = (
        arena_games.groupby(["hometeamId", "season", "arenaId"])
        .size()
        .reset_index(name="game_count")
    )

    totals = counts.groupby(["hometeamId", "season"])["game_count"].sum().rename("total")
    counts = counts.join(totals, on=["hometeamId", "season"])
    counts["pct"] = counts["game_count"] / counts["total"]

    home_arenas = counts[counts["pct"] >= _HOME_ARENA_THRESHOLD]

    result = (
        home_arenas.groupby(["hometeamId", "season"])["arenaId"]
        .apply(list)
        .reset_index()
        .rename(columns={"hometeamId": "team_id", "arenaId": "home_arena_ids"})
    )
    return result


def add_neutral_court(games: pd.DataFrame, teams_arena: pd.DataFrame) -> pd.DataFrame:
    """Add a ``neutral_court`` column to games.

    A game is neutral (1) when its ``arenaId`` is not among the home
    team's known home arenas. When ``arenaId`` is null the function falls
    back to the ``is_neutral_court_game`` column (label-based detection).

    Parameters
    ----------
    games : pd.DataFrame
        Games DataFrame. Must have ``hometeamId``, ``season``,
        ``arenaId``, and ``is_neutral_court_game``.
    teams_arena : pd.DataFrame
        Output of :func:`build_teams_arena`.

    Returns
    -------
    pd.DataFrame
        Input games with a new integer column ``neutral_court`` (0 or 1).
    """
    merged = games.merge(
        teams_arena,
        left_on=["hometeamId", "season"],
        right_on=["team_id", "season"],
        how="left",
    )

    def _flag(row) -> int:
        arena_id = row["arenaId"]
        home_arenas = row["home_arena_ids"]

        if pd.isna(arena_id) or not isinstance(home_arenas, list):
            return int(row["is_neutral_court_game"])

        return int(int(arena_id) not in home_arenas)

    merged["neutral_court"] = merged.apply(_flag, axis=1)
    return merged.drop(columns=["team_id", "home_arena_ids"])


def load_teams_arena(path) -> pd.DataFrame:
    """Load teams_arena CSV, parsing the ``home_arena_ids`` string column back to list."""
    df = pd.read_csv(path)
    df["home_arena_ids"] = df["home_arena_ids"].apply(
        lambda v: ast.literal_eval(v) if isinstance(v, str) else v
    )
    return df
