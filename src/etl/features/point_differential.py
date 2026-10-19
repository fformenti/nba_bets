"""Point differential feature calculations."""

import numpy as np
import pandas as pd

from src.etl.utils.common import has_result


def calculate_home_margin(games, lags=[], column="margin"):
    """Rolling averages of ``column`` for home teams (home games only)."""
    home_games = games.rename(
        index=str,
        columns={
            "hometeamId": "teamId",
        },
    )
    return calculate_margin(home_games, lags, column)


def calculate_away_margin(games, lags=[], column="margin"):
    """Rolling averages of ``column`` for away teams (away games only)."""
    # Work on a copy to avoid modifying the original DataFrame
    games_aux = games.copy()
    games_aux[column] = games_aux[column] * -1

    away_games = games_aux.rename(
        index=str,
        columns={
            "awayteamId": "teamId",
        },
    )

    return calculate_margin(away_games, lags, column)


def calculate_margin(games, lags=[], column="margin"):
    """
    Calculate margin statistics with rolling windows.

    Parameters
    ----------
    games : pd.DataFrame
        Games DataFrame with teamId, gameDate, season and ``column``, already
        signed from teamId's side.
    lags : list
        Window sizes; each yields ``{column}_avg_L{lag}``.
    column : str
        ``margin`` for raw points, ``margin_scaled`` for league-margin-std units.

    Returns
    -------
    pd.DataFrame
        DataFrame with margin statistics
    """
    # Sort keys must lead with the groupby keys, and gameId breaks ties so games
    # sharing a tip-off timestamp order deterministically.
    games = games.sort_values(["teamId", "season", "gameDate", "gameId"])

    games[f"{column}_L1"] = games.groupby(["teamId", "season"])[column].shift(1)

    avg_lags_cols = []
    for lag in lags:
        games[f"{column}_avg_L{lag}"] = (
            games.groupby(["teamId", "season"])[f"{column}_L1"]
            .rolling(window=lag, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        ).fillna(0.0)
        avg_lags_cols.append(f"{column}_avg_L{lag}")

    return_cols = [
        "gameId",
        "gameDate",
        "season",
        "teamId",
        column,
    ] + avg_lags_cols
    return games[return_cols].copy()


# ── Scaled margin ────────────────────────────────────────────────────────────


# The margin scale is the std of the last MARGIN_STD_WINDOW played games, across
# season boundaries. A season-bounded estimate restarted every October: it had
# to borrow last season's value for the opening weeks, then switch to a
# 100-game estimate (~7% standard error) — a 5% average step at the switch and
# the noisiest stretch of the season right after it. True drift is slow (~13 to
# ~15.7 points over three decades), so a season's worth of games loses little.
#
# The window advances by whole days: every game on a date is scaled by the
# games through the last earlier date that had any. A game never sees another
# game from its own day, whatever their tip-offs — at prediction time those are
# not final, so anything finer would make training and serving disagree.
#
# A fixed window rather than an exponential decay: prediction rebuilds features
# from the slate's two prior seasons only (src/ml/prediction/pipeline.py), and
# a window that fits inside that history reproduces the training value exactly,
# whereas a decay's tail would be cut off.
MARGIN_STD_WINDOW = 1000


def margin_std_by_date(games: pd.DataFrame) -> pd.DataFrame:
    """
    The league margin scale, one row per game date:
    ``date | last_date_with_game | margin_std``.

    ``margin_std`` is the std of the final margin (home - away) over the last
    ``MARGIN_STD_WINDOW`` played games through the end of
    ``last_date_with_game`` — the latest date strictly before ``date`` with a
    played game. Placeholder rows (upcoming games) get a row for their date but
    contribute nothing.

    This is the era unit for margins. Margin spread has drifted from ~13 to
    ~15.7 points while scoring pace moved somewhere else entirely, so it is
    measured directly rather than inferred from total points. It is the
    denominator of margin_scaled, and the LLM target is scaled by it in
    training and unscaled by it at prediction — all read this one table.

    Until W games exist — only at the very start of the history — the window
    is every earlier game; fewer than two give NaN, which _scale_margin maps to
    a margin_scaled of 0. Never backfilled from later games: that would read
    the results of the games being scaled.
    """
    played = has_result(games)
    sequence = (
        pd.DataFrame(
            {
                "date": games["gameDateOnlyStr"].to_numpy(),
                "gameId": games["gameId"].to_numpy(),
                "m": (games["homeScore"] - games["awayScore"]).to_numpy(dtype=float),
            }
        )[played.to_numpy()]
        # gameId only decides which games of the oldest day leave the window.
        .sort_values(["date", "gameId"])
    )
    m = sequence["m"].to_numpy()
    p1 = np.concatenate([[0.0], np.cumsum(m)])
    p2 = np.concatenate([[0.0], np.cumsum(m**2)])

    # Window = played games (c - W, c], c counting every game through the date.
    c = sequence.groupby("date").size().cumsum()
    lo = np.maximum(c.to_numpy() - MARGIN_STD_WINDOW, 0)
    n = c.to_numpy() - lo
    s1 = p1[c.to_numpy()] - p1[lo]
    s2 = p2[c.to_numpy()] - p2[lo]
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = (s2 - s1**2 / n) / (n - 1)
    through_date = pd.DataFrame(
        {"last_date_with_game": c.index, "margin_std": np.sqrt(np.clip(variance, 0, None))}
    )

    dates = pd.DataFrame({"date": np.sort(games["gameDateOnlyStr"].dropna().unique())})
    # Zero-padded "YYYY-MM-DD" strings, so lexical order is chronological.
    table = pd.merge_asof(
        dates.assign(_on=pd.to_datetime(dates["date"])),
        through_date.assign(_on=pd.to_datetime(through_date["last_date_with_game"])),
        on="_on",
        allow_exact_matches=False,
    )
    return table[["date", "last_date_with_game", "margin_std"]]


# ── League home-court advantage ─────────────────────────────────────────────

# Same window as MARGIN_STD_WINDOW, and for the same reason: a trailing count
# of played games large enough to be stable but fitting inside the two prior
# seasons that src/ml/prediction/pipeline.py always loads alongside a slate.
HOME_ADV_WINDOW = 1000


def home_adv_by_date(games: pd.DataFrame) -> pd.DataFrame:
    """
    League-wide home-court win rate, one row per game date:
    ``date | last_date_with_game | home_adv_L1000``.

    ``home_adv_L1000`` is the fraction of home teams that won over the last
    ``HOME_ADV_WINDOW`` played games through the end of
    ``last_date_with_game`` — the latest date strictly before ``date`` with a
    played game. Same shape and the same ``merge_asof(allow_exact_matches=
    False)`` leakage guard as ``margin_std_by_date``, just a mean rather than
    a std: every game only ever sees games through an earlier date, never its
    own day's results.
    """
    played = has_result(games)
    sequence = (
        pd.DataFrame(
            {
                "date": games["gameDateOnlyStr"].to_numpy(),
                "gameId": games["gameId"].to_numpy(),
                "home_win": (games["winner"] == games["hometeamId"]).to_numpy(
                    dtype=float
                ),
            }
        )[played.to_numpy()]
        # gameId only decides which games of the oldest day leave the window.
        .sort_values(["date", "gameId"])
    )
    w = sequence["home_win"].to_numpy()
    p1 = np.concatenate([[0.0], np.cumsum(w)])

    # Window = played games (c - W, c], c counting every game through the date.
    c = sequence.groupby("date").size().cumsum()
    lo = np.maximum(c.to_numpy() - HOME_ADV_WINDOW, 0)
    n = c.to_numpy() - lo
    s1 = p1[c.to_numpy()] - p1[lo]
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(n > 0, s1 / n, np.nan)
    through_date = pd.DataFrame(
        {"last_date_with_game": c.index, "home_adv_L1000": rate}
    )

    dates = pd.DataFrame({"date": np.sort(games["gameDateOnlyStr"].dropna().unique())})
    table = pd.merge_asof(
        dates.assign(_on=pd.to_datetime(dates["date"])),
        through_date.assign(_on=pd.to_datetime(through_date["last_date_with_game"])),
        on="_on",
        allow_exact_matches=False,
    )
    return table[["date", "last_date_with_game", "home_adv_L1000"]]


def _scale_margin(margin: pd.Series, margin_std: pd.Series) -> pd.Series:
    """
    ``margin_scaled``: margin in standard deviations of recent league margins
    (see ``margin_std_by_date``). The rolling averages read this column
    like any other, via ``calculate_margin(..., column="margin_scaled")``.
    """
    return (
        margin / margin_std.replace(0, float("nan"))
    ).fillna(0.0) + 0.0  # normalize -0.0 -> 0.0 so a later sign flip can't produce it
