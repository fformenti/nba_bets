"""Stake sizing: how many shares to buy at a given ask, given a fair price."""

from __future__ import annotations

from src.betting.polymarket_client import MarketSide, get_order_book
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def shares_for_linear_investment(fair_price: float, ask_price: float, budget: float) -> int:
    """Stake in proportion to the edge: the bigger the mispricing, the more we bet.

    The edge fraction ``(fair - ask) / ask`` scales the budget, and the result
    is converted to whole shares at the ask.
    """
    edge_fraction = round((fair_price - ask_price) / ask_price, 2)
    investment = edge_fraction * budget
    return int(investment / ask_price)


def walk_order_book(
    side: MarketSide,
    budget: float,
    advantage_threshold: float = 0.0,
) -> list[dict]:
    """Walk the asks cheapest-first, buying while the odds beat our fair odds.

    Stops at the first ask whose implied odds no longer clear
    ``advantage_threshold``. When an ask cannot absorb the full stake, it takes
    what is available, decrements the budget and continues to the next level.
    """
    order_book = get_order_book(side.token_id)
    asks = order_book["asks"]

    fair_odds = 1 / side.win_prob
    buying_strategy: list[dict] = []

    # The book arrives worst-price-first, so walk it in reverse.
    for ask in asks[::-1]:
        ask_price = float(ask["price"])
        advantage = (1 / ask_price) - fair_odds
        if advantage <= advantage_threshold:
            break

        shares_to_buy = shares_for_linear_investment(side.win_prob, ask_price, budget)
        available_shares = float(ask["size"])

        if shares_to_buy <= available_shares:
            buying_strategy.append({"ask_price": ask_price, "shares_to_buy": shares_to_buy})
            break

        # Take the whole level and carry the remaining budget to the next one.
        shares_to_buy = available_shares
        budget -= shares_to_buy * ask_price
        buying_strategy.append({"ask_price": ask_price, "shares_to_buy": shares_to_buy})

    return buying_strategy
