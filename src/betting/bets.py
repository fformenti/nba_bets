"""Turn model predictions for a date into a Polymarket buying strategy."""

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd

from src.betting.polymarket_client import get_market_sides
from src.betting.sizing import walk_order_book
from src.betting.slugs import get_game_slug
from src.config.paths import POLYMARKET_DAILY_BETS_DIR, UPCOMING_GAMES_PREDICTIONS_PATH
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

GAME_BUDGET = 100


def load_predictions_for_date(game_date_str: str) -> pd.DataFrame:
    """Predictions for one game date, with a Polymarket slug per game."""
    predictions = pd.read_csv(UPCOMING_GAMES_PREDICTIONS_PATH)

    predictions["game_slug"] = predictions.apply(
        lambda row: get_game_slug(
            row["awayteamId"], row["hometeamId"], row["gameDateOnlyStr"], row["season"]
        ),
        axis=1,
    )

    columns = [
        "gameId",
        "gameDateOnlyStr",
        "hometeamId",
        "awayteamId",
        "hometeamName",
        "awayteamName",
        "game_slug",
        "home_win_probability",
    ]
    return predictions.loc[predictions["gameDateOnlyStr"] == game_date_str, columns]


def build_daily_bets(game_date_str: str, budget: float = GAME_BUDGET) -> list[dict]:
    """Build a per-game buying strategy for every prediction on ``game_date_str``."""
    predictions = load_predictions_for_date(game_date_str)
    if predictions.empty:
        logger.warning(f"No predictions found for {game_date_str}")
        return []

    daily_bets = []
    for prediction in predictions.to_dict(orient="records"):
        game_slug = prediction["game_slug"]
        home_win_prob = round(prediction["home_win_probability"], 2)
        away_win_prob = round(1.0 - home_win_prob, 2)

        try:
            away_side, home_side = get_market_sides(game_slug)
        except Exception as exc:
            # One missing or malformed market must not abandon the rest of the slate.
            logger.warning(f"Skipping {game_slug}: {exc}")
            continue

        away_side = replace(away_side, win_prob=away_win_prob)
        home_side = replace(home_side, win_prob=home_win_prob)

        betting_strategy = []
        for side in (away_side, home_side):
            for bet in walk_order_book(side, budget):
                betting_strategy.append({**bet, "team": side.team_abv})

        logger.info(f"{game_slug}: {len(betting_strategy)} order(s)")
        daily_bets.append({"game_slug": game_slug, "betting_strategy": betting_strategy})

    return daily_bets


def save_daily_bets(daily_bets: list[dict], game_date_str: str) -> None:
    """Write the day's strategy to ``data/predictions/daily_bets/<date>.json``."""
    POLYMARKET_DAILY_BETS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = POLYMARKET_DAILY_BETS_DIR / f"{game_date_str}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(daily_bets, handle, indent=2)
    logger.info(f"Saved {len(daily_bets)} game strategies to {output_path}")


def place_bets(game_date_str: str, budget: float = GAME_BUDGET) -> list[dict]:
    """Build and persist the buying strategy for a game date."""
    daily_bets = build_daily_bets(game_date_str, budget=budget)
    save_daily_bets(daily_bets, game_date_str)
    return daily_bets
