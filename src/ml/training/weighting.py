"""Training sample weights: the within-season ramp.

One definition, called by the training path (``src/ml/training/experiment.py``).
There is no cross-season decay term — see the note above ``weighting:`` in
``configs/train/_defaults.yaml`` for the measurement that removed it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.config.schema import WeightingConfig
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def within_season_weights(metadata: pd.DataFrame, saturation_K: int) -> np.ndarray:
    """Ramp from 0 to 1 over a team's first ``saturation_K`` games of a season.

    Early-season rows carry features computed from a handful of games, so they
    are down-weighted rather than dropped. The ramp follows whichever of the two
    teams has played fewer games.
    """
    games_played = np.minimum(
        pd.to_numeric(metadata["games_played_HT"], errors="coerce").to_numpy(
            dtype=np.float64, copy=False
        ),
        pd.to_numeric(metadata["games_played_VT"], errors="coerce").to_numpy(
            dtype=np.float64, copy=False
        ),
    )
    return np.clip(games_played / float(saturation_K), 0.0, 1.0)


def compute_sample_weights(metadata: pd.DataFrame, config: WeightingConfig) -> np.ndarray | None:
    """Training weights for the rows of ``metadata``, or None if weighting is off.

    ``metadata`` must cover exactly the training rows and carry
    ``games_played_HT`` and ``games_played_VT``.
    """
    if not config.enabled:
        return None

    weights = within_season_weights(metadata, config.saturation_K)
    logger.info(
        f"Within-season weighting (K={config.saturation_K}): "
        f"min={weights.min():.2f}, mean={weights.mean():.2f}, "
        f"pct_full={(weights == 1.0).mean() * 100:.1f}%"
    )
    return weights


def effective_sample_size(weights: np.ndarray | None) -> float:
    """Kish effective sample size — how many unweighted rows the weights buy.

    Logged as ``train_effective_sample_size`` so the cost of the ramp is visible
    on the run rather than assumed.
    """
    if weights is None:
        return float("nan")
    return float(weights.sum() ** 2 / np.square(weights).sum())
