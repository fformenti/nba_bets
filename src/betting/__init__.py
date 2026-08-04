"""Polymarket integration: market lookup, slug generation, stake sizing, bets.

Kept out of ``src/ml`` because none of it is modelling — it consumes model
output (``home_win_probability``) and turns it into orders. The one shared
concept is the *slug*, which identifies a game on Polymarket and is also how
historical market odds will be joined back onto predictions.
"""

from .slugs import get_game_slug, slugify

__all__ = ["get_game_slug", "slugify"]
