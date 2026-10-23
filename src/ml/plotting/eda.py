"""EDA figures, drawn from the tables ``src/eda/`` writes.

Home win ratio by season: a bar chart over every season, and a line chart
scoped to the training population. Both read
``outputs/eda/home_win_ratio/tables/home_win_ratio_by_season.csv``.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.config.paths import (
    HOME_WIN_RATIO_BY_SEASON_CSV_PATH,
    HOME_WIN_RATIO_BY_SEASON_PNG_PATH,
    HOME_WIN_RATIO_LINE_BY_SEASON_PNG_PATH,
)

logger = logging.getLogger(__name__)

ROLLING_WINDOW = 5
# Matches configs/train/_defaults.yaml's filters.min_season - the population
# src.ml.datasets.splits.build_splits draws train/validation/test from.
MIN_SEASON = "1980/81"


def _read_win_ratio(table_path: Path) -> pd.Series:
    """``home_win_ratio`` indexed by season, in season order."""
    table = pd.read_csv(table_path).sort_values("season")
    return table.set_index("season")["home_win_ratio"]


def plot_home_win_ratio(
    table_path: Path = HOME_WIN_RATIO_BY_SEASON_CSV_PATH,
    output_path: Path = HOME_WIN_RATIO_BY_SEASON_PNG_PATH,
) -> Path:
    """Bar chart of the home-team win ratio per season, every season."""
    win_ratio = _read_win_ratio(table_path)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(16, 6))

    x = np.arange(len(win_ratio))
    seasons = win_ratio.index.tolist()

    bars = ax.bar(x, win_ratio.values, color="steelblue", alpha=0.7, width=0.6)

    rolling = win_ratio.rolling(window=ROLLING_WINDOW, center=True).mean()
    ax.plot(x, rolling.values, color="darkorange", linewidth=2, label=f"{ROLLING_WINDOW}-season rolling avg")

    ax.axhline(0.5, color="crimson", linestyle="--", linewidth=1.2, label="Coin flip (0.50)")

    for bar, val in zip(bars, win_ratio.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="dimgray",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(seasons, rotation=90, fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0.3, 0.8)

    ax.set_title("Home Team Win Ratio by NBA Season", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Season", fontsize=12)
    ax.set_ylabel("Home Win Ratio", fontsize=12)
    ax.legend(fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved plot to %s", output_path)
    return output_path


def plot_home_win_ratio_line(
    table_path: Path = HOME_WIN_RATIO_BY_SEASON_CSV_PATH,
    output_path: Path = HOME_WIN_RATIO_LINE_BY_SEASON_PNG_PATH,
) -> Path:
    """Line chart of the home-team win % per season, scoped to the training population."""
    win_ratio = _read_win_ratio(table_path)
    win_ratio = win_ratio[win_ratio.index >= MIN_SEASON]
    logger.info("Overall home win ratio (season >= %s): %.3f", MIN_SEASON, win_ratio.mean())

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(16, 6))

    x = np.arange(len(win_ratio))
    seasons = win_ratio.index.tolist()

    ax.plot(x, win_ratio.values, color="steelblue", marker="o", linewidth=1.5, label="Home win %")

    rolling = win_ratio.rolling(window=ROLLING_WINDOW, center=True).mean()
    ax.plot(x, rolling.values, color="darkorange", linewidth=2, label=f"{ROLLING_WINDOW}-season rolling avg")

    ax.axhline(0.5, color="crimson", linestyle="--", linewidth=1.2, label="Coin flip (0.50)")

    ax.set_xticks(x)
    ax.set_xticklabels(seasons, rotation=90, fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0.3, 0.8)

    ax.set_title(
        "Home Team Win % by NBA Season (train + validation + test seasons)",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Season", fontsize=12)
    ax.set_ylabel("Home Win %", fontsize=12)
    ax.legend(fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved plot to %s", output_path)
    return output_path
