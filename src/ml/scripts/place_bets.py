import requests
import json
import pandas as pd

from src.config.paths import (
    UPCOMING_GAMES_PREDICTIONS_PATH,
    POLYMARKET_DAILY_BETS_DIR,
)
from src.ml.utils.polymarket import get_game_slug

GAME_BUDGET = 100
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


def get_predictions_df(game_date_str):
    predictions_df = pd.read_csv(UPCOMING_GAMES_PREDICTIONS_PATH)

    predictions_df["game_slug"] = predictions_df.apply(
        lambda row: get_game_slug(
            row["awayteamId"], row["hometeamId"], row["gameDateOnlyStr"], row["season"]
        ),
        axis=1,
    )

    predictions_df = predictions_df[
        [
            "gameId",
            "gameDateOnlyStr",
            "hometeamId",
            "awayteamId",
            "hometeamName",
            "awayteamName",
            "game_slug",
            "home_win_probability",
        ]
    ]

    return predictions_df.loc[predictions_df["gameDateOnlyStr"] == game_date_str]


def get_market_tokens_by_slug(game_slug):
    slug_endpoint = f"{GAMMA_API}/events/slug/{game_slug}"

    events_response = requests.get(slug_endpoint)
    event_data = events_response.json()

    game_market_data = event_data["markets"][0]
    teams_names = json.loads(game_market_data["outcomes"])
    clobTokenIds = json.loads(game_market_data["clobTokenIds"])

    away_team = teams_names[0]
    home_team = teams_names[1]
    away_token_id = clobTokenIds[0]
    home_token_id = clobTokenIds[1]
    return {"team_abv": away_team, "token_id": away_token_id}, {
        "team_abv": home_team,
        "token_id": home_token_id,
    }


def find_shares_to_buy_linear_investment(fair_price, ask_price, budget):
    x = round((fair_price - ask_price) / ask_price, 2)
    investment = x * budget
    shares_to_buy = int(investment / ask_price)
    return shares_to_buy


def find_shares_to_buy_fixed_investment(ask_price, budget):
    investment = budget
    shares_to_buy = int(investment / ask_price)
    return shares_to_buy


def buy_shares(team_info, budget, advantage_threshold=0.0):
    token_id = team_info["token_id"]
    fair_prob = team_info["win_prob"]
    book_response = requests.get(f"{CLOB_API}/book", params={"token_id": token_id})
    order_book = book_response.json()
    asks = order_book["asks"]
    fair_odds = 1 / fair_prob
    buying_strategy = []
    for ask in asks[::-1]:
        ask_price = float(ask["price"])
        ask_odds = 1 / ask_price
        advantage = ask_odds - fair_odds
        if advantage > advantage_threshold:
            shares_to_buy = find_shares_to_buy_linear_investment(
                fair_prob, ask_price, budget
            )
            available_shares = float(ask["size"])
            if shares_to_buy <= available_shares:
                amount_spent = round(shares_to_buy * ask_price, 2)
                shares_bought = {
                    "ask_price": ask_price,
                    "shares_to_buy": shares_to_buy,
                }
                buying_strategy.append(shares_bought)
                break
            else:
                shares_to_buy = available_shares
                amount_spent = shares_to_buy * ask_price
                shares_bought = {
                    "ask_price": ask_price,
                    "shares_to_buy": shares_to_buy,
                }
                budget -= amount_spent
                buying_strategy.append(shares_bought)
        else:
            break
    return buying_strategy


def place_bets(game_date_str):
    predictions_date = get_predictions_df(game_date_str)

    games_slugs_preds = predictions_date[
        ["gameId", "game_slug", "home_win_probability"]
    ].to_dict(orient="records")
    daily_bets = []
    for game_slug_pred in games_slugs_preds:
        print(game_slug_pred)
        betting_strategy = []
        game_slug = game_slug_pred["game_slug"]
        home_win_prob = round(game_slug_pred["home_win_probability"], 2)
        away_win_prob = round(1.0 - home_win_prob, 2)

        away_team_info, home_team_info = get_market_tokens_by_slug(game_slug)
        away_bets = buy_shares(
            away_team_info["away_token_id"], away_win_prob, GAME_BUDGET
        )

        home_bets = buy_shares(
            home_team_info["home_token_id"], home_win_prob, GAME_BUDGET
        )
        if len(away_bets) > 0:
            for bet in away_bets:
                bet["team"] = away_team_info["away_team"]
            betting_strategy.extend(away_bets)
        if len(home_bets) > 0:
            for bet in home_bets:
                bet["team"] = home_team_info["home_team"]
            betting_strategy.extend(home_bets)

        daily_bets.append(
            {"game_slug": game_slug, "betting_strategy": betting_strategy}
        )

    return daily_bets


def save_daily_bets(daily_bets, game_date_str):
    # create the directory if it doesn't exist
    POLYMARKET_DAILY_BETS_DIR.mkdir(parents=True, exist_ok=True)
    with open(POLYMARKET_DAILY_BETS_DIR / f"{game_date_str}.json", "w") as f:
        json.dump(daily_bets, f)


if __name__ == "__main__":
    game_date_str = "2026-02-09"
    daily_bets = place_bets(game_date_str)
    save_daily_bets(daily_bets, game_date_str)
