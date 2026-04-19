"""Polymarket slug generation utilities."""

import pandas as pd
from functools import lru_cache

from src.config.paths import POLYMARKET_TEAMS_ABV_PATH


def slugify(away_abv: str, home_abv: str, game_date: str) -> str:
    """Format a Polymarket game slug from team abbreviations and date."""
    return f"nba-{away_abv.lower()}-{home_abv.lower()}-{game_date}"


@lru_cache(maxsize=None)
def _load_team_abv_lookup() -> dict:
    """Load {teamId: polymarket_abv} from the processed CSV. Cached after first call.

    Season is intentionally ignored — no NBA franchises have been created or
    relocated since 2021, so the 2025/26 abbreviations apply to all historical games.
    """
    df = pd.read_csv(POLYMARKET_TEAMS_ABV_PATH)
    return {int(row.teamId): row.polymarket_abv for row in df.itertuples()}


def get_game_slug(away_team_id: int, home_team_id: int, game_date: str, season: str = "") -> str:
    """Generate a Polymarket game slug from team IDs. Works on any game record.

    The season parameter is kept for backward compatibility but is not used for
    the abbreviation lookup.
    """
    lookup = _load_team_abv_lookup()
    away_abv = lookup[int(away_team_id)]
    home_abv = lookup[int(home_team_id)]
    return slugify(away_abv, home_abv, game_date)
