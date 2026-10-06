"""Polymarket slug generation for the bet-placement path.

A slug (``nba-bos-lal-2026-02-09``) identifies a game's Polymarket market. This
builds one for an *upcoming* game, where no market listing has been fetched yet.

The historical backfill does not use this and should not: it discovers real
slugs from Gamma's event listing rather than guessing them
(:mod:`src.betting.polymarket_odds`). A guessed slug has to be right about
things that turned out to vary — team order flipped between 2023/24 and 2024/25,
an optional ``-moneyline`` suffix appeared in March 2025, and five teams changed
abbreviation. For an upcoming game none of that applies: the current codes and
the away-first order are what Polymarket has used since 2024-10-28.
"""

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

    One abbreviation per team, the current one. Five teams did use a different
    code once (``gs``, ``no``, ``sa``, ``utah``, ``wsh``), but not once after
    2024-10-28, so a slug built for an upcoming game is never affected. Games
    from that era are reached through the odds backfill's listing instead.
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
