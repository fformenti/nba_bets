"""EDA: Home team win ratio by NBA season - the table.

Writes ``outputs/eda/home_win_ratio/tables/home_win_ratio_by_season.csv``. The
two figures are drawn from that file by :mod:`src.ml.plotting.eda`, so the
numbers behind them can be read without the chart.
"""

import logging
from pathlib import Path

import pandas as pd

from src.config.paths import HOME_WIN_RATIO_BY_SEASON_CSV_PATH, REGULAR_SEASON_GAMES_PATH

logger = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(REGULAR_SEASON_GAMES_PATH)
    logger.info("Loaded %d games across %d seasons", len(df), df["season"].nunique())
    return df


def compute_home_win_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """One row per season: ``season``, ``n_games``, ``home_wins``, ``home_win_ratio``."""
    home_win = (df["winner"] == df["hometeamId"]).astype(int)
    table = (
        home_win.groupby(df["season"])
        .agg(n_games="size", home_wins="sum")
        .reset_index()
        .sort_values("season")
    )
    table["home_win_ratio"] = table["home_wins"] / table["n_games"]
    return table


def write_home_win_ratio_table(path: Path = HOME_WIN_RATIO_BY_SEASON_CSV_PATH) -> Path:
    """Compute the per-season home win ratio from the game table and save it."""
    table = compute_home_win_ratio(load_data())
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    logger.info("Overall home win ratio: %.3f", table["home_win_ratio"].mean())
    logger.info("Table saved to %s", path)
    return path
