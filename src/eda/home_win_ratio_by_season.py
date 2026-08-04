"""EDA: Home team win ratio by NBA season."""

import logging

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.config.paths import HOME_WIN_RATIO_BY_SEASON_PNG_PATH, REGULAR_SEASON_GAMES_PATH

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROLLING_WINDOW = 5


def load_data() -> pd.DataFrame:
    df = pd.read_csv(REGULAR_SEASON_GAMES_PATH)
    logger.info("Loaded %d games across %d seasons", len(df), df["season"].nunique())
    return df


def compute_home_win_ratio(df: pd.DataFrame) -> pd.Series:
    home_win = df["winner"] == df["hometeamId"]
    return home_win.groupby(df["season"]).mean()


def plot(win_ratio: pd.Series) -> None:
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
    HOME_WIN_RATIO_BY_SEASON_PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(HOME_WIN_RATIO_BY_SEASON_PNG_PATH, dpi=300)
    logger.info("Saved plot to %s", HOME_WIN_RATIO_BY_SEASON_PNG_PATH)


def plot_home_win_ratio() -> None:
    """Chart the home-team win ratio per season."""
    df = load_data()
    win_ratio = compute_home_win_ratio(df)
    logger.info("Overall home win ratio: %.3f", win_ratio.mean())
    plot(win_ratio)
