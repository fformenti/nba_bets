"""Score the model against what Polymarket priced at tip-off.

The benchmark question: on the games where both had an opinion, did the model
beat the market? Both sides are scored by
:func:`src.ml.evaluation.metrics.compute_home_win_metrics` — the same function
the live scorecard uses — so any gap between them comes from the predictions and
not from two different notions of accuracy.

**Mutual coverage is the whole point.** A game Polymarket never listed, or one
whose market never traded before tip-off, is dropped from *both* sides rather
than from one. Scoring the model on 1,200 games and the market on 900 would
compare two different questions and flatter whichever side got the easier slate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config.paths import INGESTED_POLYMARKET_ODDS_PATH
from src.ml.evaluation.metrics import compute_home_win_metrics
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Markets below this dollar volume are quotes in name only: their price series
# sits flat at 0.5 for the life of the market, which would enter the benchmark
# as a "coin flip" opinion the market never actually held.
#
# It excludes 8 of 1,913 games and moves market accuracy by 0.0003, so it is a
# guard rather than a correction. It used to matter more — before duplicate
# markets were resolved by volume, the near-empty `-moneyline` twins pushed this
# to 17 games and 12 flat prices.
MIN_VOLUME = 1000.0


def load_market_probabilities(
    path: Path = INGESTED_POLYMARKET_ODDS_PATH,
    min_volume: float = MIN_VOLUME,
) -> pd.DataFrame:
    """``gameId → market_home_win_probability`` for every usably-priced game.

    Returns only resolved rows: ``not_found == 0``, both sides priced, and
    volume at or above ``min_volume``.

    The two sides are near-exact complements — over a full backfill, 1,912 of
    1,913 pairs summed to exactly 1.0000 and the outlier to 0.9950 — so
    normalising by the pair is a guard against a malformed row rather than the
    removal of a spread. It is kept because it costs nothing and because a row
    whose sides *don't* sum to one is precisely the row worth not trusting.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No Polymarket odds at {path}. Run `make build-polymarket-odds` first."
        )

    odds = pd.read_csv(path)
    total = len(odds)

    odds = odds[odds["not_found"] == 0]
    priced = odds["home_odds"] + odds["away_odds"]
    odds = odds[(priced > 0) & (odds["volume"] >= min_volume)]

    odds = odds.assign(
        market_home_win_probability=odds["home_odds"] / (odds["home_odds"] + odds["away_odds"])
    )
    odds["gameId"] = pd.to_numeric(odds["gameId"], errors="coerce").astype("Int64")

    logger.info(
        f"{len(odds)}/{total} Polymarket row(s) usable "
        f"(resolved, priced, volume >= {min_volume:,.0f})"
    )
    return odds[["gameId", "market_home_win_probability", "volume"]]


def align_market_to_rows(
    game_ids: pd.Series,
    market: Optional[pd.DataFrame] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Which of ``game_ids`` the market priced, and those prices in row order.

    ``game_ids`` is ``metadata_test["gameId"]``. The returned mask is positional,
    so the same mask slices ``y_test``, ``y_test_proba``, ``uncalibrated_proba``
    and ``metadata_test`` — every test-split array is row-aligned with the
    metadata. That is what lets a plot put the model and the market on identical
    games rather than on two slates of different size.
    """
    if market is None:
        market = load_market_probabilities()

    market = market.drop_duplicates(subset="gameId")
    prices = pd.Series(
        market["market_home_win_probability"].to_numpy(),
        index=pd.Index(market["gameId"]),
    )

    ids = pd.to_numeric(pd.Series(game_ids), errors="coerce").astype("Int64")
    mask = ids.isin(prices.index).to_numpy()
    return mask, prices.reindex(ids[mask]).to_numpy()


def compare_to_market(
    scored: pd.DataFrame,
    market: Optional[pd.DataFrame] = None,
    scope: str = "overall",
) -> pd.DataFrame:
    """Model and market metrics over the games both of them cover.

    ``scored`` needs ``gameId``, ``actual_home_win``, ``prediction`` and
    ``home_win_probability`` — the shape
    :func:`src.monitoring.scoring.join_predictions_to_outcomes` already
    produces, and the shape :func:`build_scored_frame` builds from a backtest.

    Returns one row per source with an identical ``n_games``. If they ever
    differ, the mutual-coverage filter has been bypassed.
    """
    if market is None:
        market = load_market_probabilities()

    scored = scored.copy()
    scored["gameId"] = pd.to_numeric(scored["gameId"], errors="coerce").astype("Int64")

    mutual = scored.merge(market, on="gameId", how="inner")
    dropped = len(scored) - len(mutual)
    if mutual.empty:
        logger.warning("No games are covered by both the model and Polymarket; nothing to compare.")
        return pd.DataFrame()

    # The market's hard call is just its favourite.
    mutual["market_prediction"] = (mutual["market_home_win_probability"] >= 0.5).astype(int)

    model_metrics = compute_home_win_metrics(mutual)
    market_metrics = compute_home_win_metrics(
        mutual,
        prediction_column="market_prediction",
        probability_column="market_home_win_probability",
    )

    logger.info(
        f"Compared {len(mutual)} game(s) against Polymarket "
        f"({dropped} dropped for lack of mutual coverage): "
        f"model {model_metrics['accuracy']:.1%} vs market {market_metrics['accuracy']:.1%}"
    )

    return pd.DataFrame(
        [
            {"scope": scope, "source": "model", "n_dropped": dropped, **model_metrics},
            {"scope": scope, "source": "polymarket", "n_dropped": dropped, **market_metrics},
        ]
    )


def build_scored_frame(
    game_ids: pd.Series,
    y_true,
    y_pred,
    y_pred_proba=None,
) -> pd.DataFrame:
    """Assemble a backtest's test split into the shape :func:`compare_to_market` wants.

    The training run holds its test-set predictions only in memory, aligned by
    position with ``metadata_test``; this pairs them back up with their gameIds
    so they can be joined to market prices.
    """
    frame = pd.DataFrame(
        {
            "gameId": pd.Series(game_ids).to_numpy(),
            "actual_home_win": pd.Series(y_true).to_numpy(),
            "prediction": pd.Series(y_pred).to_numpy(),
        }
    )
    if y_pred_proba is not None:
        proba = pd.Series(y_pred_proba).to_numpy()
        frame["home_win_probability"] = proba
    return frame
