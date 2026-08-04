"""Live model performance: how predictions actually did once games were played.

Distinct from ``src/ml/evaluation``, which scores a model against a frozen
holdout at training time. This package scores the predictions the pipeline
really emitted, against the games that really happened — the feedback half of
the predict → play → score → fold-into-history loop.
"""

from .scoring import score_predictions

__all__ = ["score_predictions"]
