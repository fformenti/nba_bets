"""
Freeze the holdout (test) set from games_features.csv.

Run once to define the fixed evaluation boundary. All future training runs
will evaluate against this same set of games, enabling stable model comparisons
and external benchmarking (e.g. vs Polymarket consensus).

Usage:
    uv run python -m src.data_creation.make_holdout_set
    uv run python -m src.data_creation.make_holdout_set --holdout-start-season 2023/24
"""

import argparse
import logging

import pandas as pd

from src.config.paths import HOLDOUT_DIR, HOLDOUT_TEST_METADATA_PATH, REGULAR_SEASON_GAMES_FEATURES_PATH

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

METADATA_COLS = ["gameId", "season", "gameDateOnlyStr", "hometeamId", "awayteamId"]
DEFAULT_MIN_SEASON = "1980/81"
DEFAULT_HOLDOUT_START_SEASON = "2022/23"


def make_holdout_set(holdout_start_season: str, min_season: str) -> None:
    df = pd.read_csv(
        REGULAR_SEASON_GAMES_FEATURES_PATH,
        usecols=lambda c: c in METADATA_COLS + ["gameDate"],
        low_memory=False,
    )

    # Apply the same min_season filter used during training
    df = df[df["season"] >= min_season].copy()

    holdout = df[df["season"] >= holdout_start_season][METADATA_COLS].copy()
    holdout = holdout.drop_duplicates(subset=["gameId"])

    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    holdout.to_csv(HOLDOUT_TEST_METADATA_PATH, index=False)

    seasons = sorted(holdout["season"].unique())
    logger.info(
        f"Holdout set frozen: {len(holdout)} games | "
        f"seasons {seasons[0]} – {seasons[-1]} | "
        f"saved to {HOLDOUT_TEST_METADATA_PATH}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freeze the holdout test set.")
    parser.add_argument(
        "--holdout-start-season",
        default=DEFAULT_HOLDOUT_START_SEASON,
        help=f"First season included in the holdout (default: {DEFAULT_HOLDOUT_START_SEASON})",
    )
    parser.add_argument(
        "--min-season",
        default=DEFAULT_MIN_SEASON,
        help=f"Earliest season kept before splitting (default: {DEFAULT_MIN_SEASON})",
    )
    args = parser.parse_args()
    make_holdout_set(args.holdout_start_season, args.min_season)
