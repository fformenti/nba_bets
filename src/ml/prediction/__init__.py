"""Prediction pipeline for upcoming NBA games.

``pipeline.run_prediction_pipeline`` is the entry point; ``features`` mirrors
the training-time feature construction and ``io`` loads the fetched
upcoming-game JSON.
"""

from .pipeline import run_prediction_pipeline

__all__ = ["run_prediction_pipeline"]
