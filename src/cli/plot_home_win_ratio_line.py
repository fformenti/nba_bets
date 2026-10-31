"""Line-chart the home-team win % per season, scoped to the training population.

Writes the per-season table first, then draws the line chart from it
(outputs/eda/home_win_ratio/; see docs/reference/outputs.md).
"""

import matplotlib

# Files only, never a window.
matplotlib.use("Agg")

from src.eda.home_win_ratio_by_season import write_home_win_ratio_table  # noqa: E402
from src.ml.plotting.eda import plot_home_win_ratio_line  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402


def main() -> None:
    setup_logging(level="INFO")
    plot_home_win_ratio_line(write_home_win_ratio_table())


if __name__ == "__main__":
    main()
