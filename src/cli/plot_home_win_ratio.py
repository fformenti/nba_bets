"""Chart the home-team win ratio per season."""

from src.eda.home_win_ratio_by_season import plot_home_win_ratio
from src.utils.logging_config import setup_logging


def main() -> None:
    setup_logging(level="INFO")
    plot_home_win_ratio()


if __name__ == "__main__":
    main()
