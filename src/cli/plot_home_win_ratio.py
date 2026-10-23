"""Chart the home-team win ratio per season.

Writes the per-season table first, then draws the bar chart from it
(outputs/eda/home_win_ratio/; see docs/OUTPUTS.md).
"""

import matplotlib

# Files only, never a window.
matplotlib.use("Agg")

from src.eda.home_win_ratio_by_season import write_home_win_ratio_table  # noqa: E402
from src.ml.plotting.eda import plot_home_win_ratio  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402


def main() -> None:
    setup_logging(level="INFO")
    plot_home_win_ratio(write_home_win_ratio_table())


if __name__ == "__main__":
    main()
