"""EDA writes a table; the figures are drawn from that table alone."""

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from src.eda.home_win_ratio_by_season import compute_home_win_ratio
from src.ml.plotting.eda import MIN_SEASON, plot_home_win_ratio, plot_home_win_ratio_line


def test_table_counts_home_wins_per_season():
    games = pd.DataFrame(
        {
            "season": ["1979/80", "1979/80", "1980/81", "1980/81", "1980/81"],
            "hometeamId": [1, 2, 1, 2, 3],
            # `winner` is a teamId, not a flag.
            "winner": [1, 9, 1, 2, 9],
        }
    )
    table = compute_home_win_ratio(games)

    assert table.to_dict(orient="records") == [
        {"season": "1979/80", "n_games": 2, "home_wins": 1, "home_win_ratio": 0.5},
        {"season": "1980/81", "n_games": 3, "home_wins": 2, "home_win_ratio": 2 / 3},
    ]


def test_figures_draw_from_the_table_file(tmp_path):
    seasons = [f"{y}/{str(y + 1)[-2:]}" for y in range(1975, 1990)]
    table_path = tmp_path / "table.csv"
    pd.DataFrame(
        {"season": seasons, "n_games": 100, "home_wins": 60, "home_win_ratio": 0.6}
    ).to_csv(table_path, index=False)

    bar = plot_home_win_ratio(table_path, tmp_path / "bar.png")
    line = plot_home_win_ratio_line(table_path, tmp_path / "line.png")

    assert bar.exists() and line.exists()
    assert MIN_SEASON in seasons
