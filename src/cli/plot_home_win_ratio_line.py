"""Line-chart the home-team win % per season, scoped to the training population."""

from src.eda.home_win_ratio_by_season import plot_home_win_ratio_line
from src.utils.logging_config import setup_logging


def main() -> None:
    setup_logging(level="INFO")
    plot_home_win_ratio_line()


if __name__ == "__main__":
    main()
