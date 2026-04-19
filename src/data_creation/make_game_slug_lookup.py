"""
Generate a game_id → game_slug lookup table from the frozen holdout set.

This lookup is used to enrich the test dataset with Polymarket market slugs
so that win probability data can be fetched for each game in a later step.

Usage:
    uv run python -m src.data_creation.make_game_slug_lookup
"""

import logging

import pandas as pd

from src.config.paths import GAME_SLUG_LOOKUP_PATH, HOLDOUT_TEST_METADATA_PATH, PROCESSED_DIR
from src.ml.utils.polymarket import get_game_slug

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def make_game_slug_lookup() -> None:
    if not HOLDOUT_TEST_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Holdout metadata not found at {HOLDOUT_TEST_METADATA_PATH}. "
            "Run `make make-holdout-set` first."
        )

    df = pd.read_csv(HOLDOUT_TEST_METADATA_PATH)

    df["game_slug"] = df.apply(
        lambda row: get_game_slug(
            row["awayteamId"],
            row["hometeamId"],
            row["gameDateOnlyStr"],
            row["season"],
        ),
        axis=1,
    )

    lookup = df[["gameId", "game_slug"]].rename(columns={"gameId": "game_id"})

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    lookup.to_csv(GAME_SLUG_LOOKUP_PATH, index=False)

    logger.info(
        f"Game slug lookup saved: {len(lookup)} games → {GAME_SLUG_LOOKUP_PATH}"
    )
    logger.info(f"Sample:\n{lookup.head(3).to_string(index=False)}")


if __name__ == "__main__":
    make_game_slug_lookup()
