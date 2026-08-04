"""Freeze the holdout (test) set from games_features.csv.

Run once to define the fixed evaluation boundary. Every training run — sklearn
and LLM alike — then evaluates on this same set of games, which is what makes
model comparisons stable and enables external benchmarking (e.g. against the
Polymarket favourite).

Consumed by :func:`src.ml.datasets.splits.build_splits`. Run via
``make build-holdout-set``.
"""

import pandas as pd

from src.config.paths import (
    HOLDOUT_DIR,
    HOLDOUT_TEST_METADATA_PATH,
    REGULAR_SEASON_GAMES_FEATURES_PATH,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

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
