"""Generate a game_id → game_slug lookup table for the test seasons.

The lookup is the join key between our test games and Polymarket markets, so
market odds can be pulled per game and compared against model predictions.

Which games qualify is not decided here: it is ``splitting.test_start_season``
from the shared training defaults, the same value
:func:`src.ml.datasets.splits.build_splits` uses to carve out the test set. One
declaration, so the slugs can never cover a different set of games than the
models are scored on.

Run via ``make build-game-slug-lookup``.
"""

from typing import Optional

import pandas as pd

from src.betting.slugs import get_game_slug
from src.config.paths import (
    GAME_SLUG_LOOKUP_PATH,
    PROCESSED_DIR,
    REGULAR_SEASON_GAMES_FEATURES_PATH,
    TRAIN_DEFAULTS_CONFIG_PATH,
)
from src.ml.config.loader import load_experiment_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

COLUMNS = [
    "gameId",
    "season",
    "gameDate",
    "gameDateOnlyStr",
    "hometeamId",
    "awayteamId",
    "hometeamName",
    "awayteamName",
]


def load_games_since(start_season: Optional[str] = None) -> pd.DataFrame:
    """Games from ``start_season`` on, one row per gameId.

    ``start_season`` defaults to ``splitting.test_start_season`` — the same
    value :func:`src.ml.datasets.splits.build_splits` uses to carve out the test
    set, so the default can never cover a different set of games than the models
    are scored on. The Polymarket odds backfill passes an earlier season
    explicitly, because a market benchmark is worth having over more history
    than the holdout.

    Seasons compare as strings, which is correct for the zero-padded
    ``YYYY/YY`` form and matches how ``season_split`` masks the same column.
    """
    if start_season is None:
        config = load_experiment_config(TRAIN_DEFAULTS_CONFIG_PATH)
        start_season = config.splitting.test_start_season
        if start_season is None:
            raise ValueError(
                f"splitting.test_start_season is unset in {TRAIN_DEFAULTS_CONFIG_PATH}, "
                "so there is no test-season boundary to build slugs for."
            )

    df = pd.read_csv(
        REGULAR_SEASON_GAMES_FEATURES_PATH,
        usecols=lambda c: c in COLUMNS,
        low_memory=False,
    )
    df = df[df["season"] >= start_season].drop_duplicates(subset=["gameId"]).copy()

    if df.empty:
        raise ValueError(
            f"No games on or after season '{start_season}' in "
            f"{REGULAR_SEASON_GAMES_FEATURES_PATH}. Run `make build-features` first."
        )
    return df


def make_game_slug_lookup() -> None:
    df = load_games_since()
    test_start_season = df["season"].min()

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
        f"Game slug lookup saved: {len(lookup)} games from {test_start_season} on "
        f"→ {GAME_SLUG_LOOKUP_PATH}"
    )
    logger.info(f"Sample:\n{lookup.head(3).to_string(index=False)}")
