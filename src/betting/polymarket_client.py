"""HTTP access to Polymarket's Gamma (events) and CLOB (order book) APIs.

The only module that talks to Polymarket over the network. Everything else in
``src/betting`` works on the dataclasses returned here, so sizing and bet
construction stay testable without a live market.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import requests

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

# Polymarket occasionally stalls rather than refusing; without this a bet run
# can hang indefinitely mid-slate.
REQUEST_TIMEOUT_SECONDS = 15


@dataclass
class MarketSide:
    """One side of a game market: the team, its CLOB token, and our fair price.

    ``win_prob`` is the model's probability for this side, attached by the
    caller. Bundling it with the token means sizing functions receive a single
    typed object instead of a loose dict — the previous shape mismatch between
    ``get_market_tokens_by_slug`` and ``buy_shares`` silently broke every bet
    run.
    """

    team_abv: str
    token_id: str
    win_prob: float = 0.0


def get_market_sides(game_slug: str) -> tuple[MarketSide, MarketSide]:
    """Return ``(away, home)`` sides for a game, by Polymarket event slug.

    Polymarket orders both ``outcomes`` and ``clobTokenIds`` away-team-first.
    """
    response = requests.get(f"{GAMMA_API}/events/slug/{game_slug}", timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    event_data = response.json()

    markets = event_data.get("markets") or []
    if not markets:
        raise ValueError(f"No markets on Polymarket event '{game_slug}'")

    market = markets[0]
    team_abvs = json.loads(market["outcomes"])
    token_ids = json.loads(market["clobTokenIds"])

    away = MarketSide(team_abv=team_abvs[0], token_id=token_ids[0])
    home = MarketSide(team_abv=team_abvs[1], token_id=token_ids[1])
    return away, home


def get_order_book(token_id: str) -> dict:
    """Fetch the CLOB order book for one outcome token."""
    response = requests.get(
        f"{CLOB_API}/book",
        params={"token_id": token_id},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def get_market_prices(game_slug: str) -> Optional[dict]:
    """Current mid prices per outcome for a game, as ``{team_abv: price}``.

    TODO: this is the seam for backfilling *historical* market odds so model
    accuracy can be compared against the Polymarket favourite. The join key is
    ``game_slug``, which ``src/betting/game_slugs.py`` already materialises for
    every holdout game as ``data/processed/game_slug_lookup.csv``.

    Gamma's ``/events/slug/{slug}`` returns ``outcomePrices`` alongside
    ``outcomes`` for settled markets, so a backfill can read closing prices
    straight from this endpoint. Returns ``None`` when the market is unknown.
    """
    response = requests.get(f"{GAMMA_API}/events/slug/{game_slug}", timeout=REQUEST_TIMEOUT_SECONDS)
    if response.status_code == 404:
        logger.warning(f"No Polymarket event for slug '{game_slug}'")
        return None
    response.raise_for_status()

    markets = response.json().get("markets") or []
    if not markets:
        return None

    market = markets[0]
    if "outcomePrices" not in market:
        return None

    team_abvs = json.loads(market["outcomes"])
    prices = [float(p) for p in json.loads(market["outcomePrices"])]
    return dict(zip(team_abvs, prices, strict=True))
