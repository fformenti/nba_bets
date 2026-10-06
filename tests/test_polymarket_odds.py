"""Polymarket odds backfill: market identification, matching, market baseline.

Everything here runs against stubbed markets. Nothing touches the network — the
point of the client/builder split is that matching and orientation logic stays
testable without a live market, the same way `test_bet_selection.py` tests bet
sizing without one.

The fixtures are shaped like real Gamma responses, including the parts that are
wrong: a moneyline mislabelled `totals`, and a game carrying two markets where
one of them never traded.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.betting import polymarket_client, polymarket_odds
from src.betting.polymarket_client import is_game_market
from src.ml.evaluation import polymarket_baseline

WARRIORS = 1610612744
SUNS = 1610612756
LAKERS = 1610612747

NICKNAMES = frozenset(
    {"Warriors", "Suns", "Lakers", "Clippers", "Raptors", "Bucks", "Thunder", "Nuggets"}
)


@pytest.fixture(autouse=True)
def nicknames(monkeypatch):
    """Pin the team vocabulary so nothing calls Gamma's /teams endpoint."""
    monkeypatch.setattr(polymarket_client, "nba_team_nicknames", lambda: NICKNAMES)


# ---------------------------------------------------------------------------
# Which markets count as a game
# ---------------------------------------------------------------------------


def test_moneyline_is_a_game_market():
    assert is_game_market(
        {
            "slug": "nba-lal-gsw-2024-12-25",
            "sportsMarketType": "moneyline",
            "outcomes": '["Lakers","Warriors"]',
        }
    )


def test_the_moneyline_suffix_is_still_a_game_market():
    """Polymarket started appending `-moneyline` to some slugs in March 2025."""
    assert is_game_market(
        {
            "slug": "nba-bos-mia-2025-03-14-moneyline",
            "sportsMarketType": "moneyline",
            "outcomes": '["Raptors","Bucks"]',
        }
    )


def test_a_moneyline_mislabelled_totals_is_still_a_game():
    """`nba-okc-den-2024-10-24` is typed "totals" and is the only market on its event.

    49 markets in 2024/25 carry this bug, 34 of them Thunder away games. Trusting
    `sportsMarketType` drops them; trusting the slug does not.
    """
    assert is_game_market(
        {
            "slug": "nba-okc-den-2024-10-24",
            "sportsMarketType": "totals",
            "outcomes": '["Thunder","Nuggets"]',
        }
    )


def test_a_spread_market_is_rejected_by_its_slug():
    """Spreads carry two team nicknames too — only the slug suffix separates them."""
    assert not is_game_market(
        {
            "slug": "nba-tor-bkn-2025-03-26-spread-away-3pt5",
            "sportsMarketType": "spreads",
            "outcomes": '["Raptors","Bucks"]',
        }
    )


def test_a_future_that_looks_like_a_game_slug_is_rejected_by_its_outcomes():
    assert not is_game_market(
        {"slug": "nba-lebron-james-next-team", "outcomes": '["Yes","No"]'}
    )


def test_an_all_star_market_is_not_a_game():
    """Gamma's NBA team list includes 7 All-Star squads; none play real games."""
    assert not is_game_market(
        {
            "slug": "nba-sog-cgs-2025-02-16",
            "sportsMarketType": "moneyline",
            "outcomes": '["Shaq\'s OGs","Chuck\'s Global Stars"]',
        }
    )


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------


def _game(home_name="Warriors", away_name="Suns"):
    """One of our game rows, as itertuples yields it."""
    frame = pd.DataFrame(
        [
            {
                "gameId": 22300001,
                "gameDate": "2023-10-24 22:00:00",
                "gameDateOnlyStr": "2023-10-24",
                "season": "2023/24",
                "hometeamId": WARRIORS,
                "awayteamId": SUNS,
                "hometeamName": home_name,
                "awayteamName": away_name,
            }
        ]
    )
    return next(frame.itertuples())


# A real 2023/24 market: home team first in both the slug and the outcomes.
HOME_FIRST_MARKET = {
    "slug": "nba-gs-phx-2023-10-24",
    "conditionId": "0xabc",
    "clobTokenIds": '["token-warriors","token-suns"]',
    "outcomes": '["Warriors","Suns"]',
    "gameStartTime": "2023-10-25 02:00:00+00",
    "volume": "12345.6",
    "sportsMarketType": None,
}


def test_home_away_come_from_our_record_not_the_market_order(monkeypatch):
    """The 2023/24 slug lists the home team first; the row must still be right.

    Reading `outcomes[0]` as the away team — which is correct from 2024/25 on —
    would invert every 2023/24 game, and the error is invisible: the odds still
    look like plausible odds.
    """
    monkeypatch.setattr(
        polymarket_odds,
        "_side_price",
        lambda tokens, index, when: (
            (0.7, pd.Timestamp("2023-10-25T01:00:00Z"), "prices_history")
            if index == 0
            else (0.3, pd.Timestamp("2023-10-25T01:00:00Z"), "prices_history")
        ),
    )

    row = polymarket_odds._build_row(_game(), HOME_FIRST_MARKET)

    assert row["not_found"] == 0
    # Warriors are home for us and sit at outcomes[0], priced 0.7.
    assert row["home_team"] == "gs"
    assert row["away_team"] == "phx"
    assert row["home_odds"] == 0.7
    assert row["away_odds"] == 0.3


def test_a_market_for_the_wrong_teams_is_not_a_match(monkeypatch):
    """A market whose outcomes do not contain our home team resolves to nothing."""
    row = polymarket_odds._build_row(_game(home_name="Lakers", away_name="Clippers"),
                                     HOME_FIRST_MARKET)

    assert row["not_found"] == 1
    assert row["gameId"] == 22300001


def test_a_market_that_never_traded_keeps_its_slug_but_zeroes_the_odds(monkeypatch):
    """Untraded 2023/24 markets are a finding about the market, not a missing game."""
    monkeypatch.setattr(
        polymarket_odds, "_side_price", lambda tokens, index, when: (None, None, "none")
    )

    row = polymarket_odds._build_row(_game(), HOME_FIRST_MARKET)

    assert row["not_found"] == 1
    assert row["home_odds"] == 0.0 and row["away_odds"] == 0.0
    assert row["home_qty"] == 0 and row["away_qty"] == 0
    # ...but we still know which market it was, and that nobody traded it.
    assert row["game_slug"] == "nba-gs-phx-2023-10-24"
    assert row["volume"] == pytest.approx(12345.6)


def test_a_priced_game_carries_placeholder_depth(monkeypatch):
    """No order book survives a market's close, so qty is a marker, not a size."""
    monkeypatch.setattr(
        polymarket_odds,
        "_side_price",
        lambda tokens, index, when: (
            0.5,
            pd.Timestamp("2023-10-25T01:00:00Z"),
            "prices_history",
        ),
    )

    row = polymarket_odds._build_row(_game(), HOME_FIRST_MARKET)

    assert row["home_qty"] == polymarket_odds.PLACEHOLDER_QTY == 1
    assert row["away_qty"] == 1


def test_unresolved_row_keeps_the_original_game_id():
    row = polymarket_odds._unresolved_row(22300099)

    assert row["gameId"] == 22300099
    assert row["not_found"] == 1
    assert row["home_odds"] == 0.0 and row["away_qty"] == 0
    assert set(row) == set(polymarket_odds.COLUMNS)


# ---------------------------------------------------------------------------
# Tip-off dates
# ---------------------------------------------------------------------------


def test_a_late_tipoff_keeps_the_eastern_calendar_date():
    """A 10pm Eastern game is already tomorrow in UTC.

    Keying markets on the UTC date would misfile every evening game by a day —
    which is all of them.
    """
    assert polymarket_odds._eastern_date("2023-10-25 02:00:00+00") == pd.Timestamp(
        "2023-10-24"
    ).date()


# ---------------------------------------------------------------------------
# Matching games to markets
# ---------------------------------------------------------------------------


def test_an_unlisted_game_resolves_to_nothing():
    """The listing is authoritative; there is no slug-guessing second chance."""
    assert polymarket_odds._resolve_market(_game(), index={}) is None


def test_a_game_matches_on_its_matchup_and_date():
    index = {
        (frozenset({"Warriors", "Suns"}), pd.Timestamp("2023-10-24").date()): HOME_FIRST_MARKET
    }

    assert polymarket_odds._resolve_market(_game(), index) is HOME_FIRST_MARKET


def test_a_market_is_filed_under_both_its_slug_and_tipoff_dates():
    """`nba-dal-uta-2024-11-14` tips off at 05:57 UTC — 00:57 Eastern, the 15th.

    The slug, the game and our gameDateOnlyStr all say the 14th, so keying on the
    Eastern tip-off alone would miss it. Filing under both dates catches every
    game without a ±1-day fudge that could grab a back-to-back.
    """
    market = {
        "slug": "nba-dal-uta-2024-11-14",
        "outcomes": '["Mavericks","Jazz"]',
        "gameStartTime": "2024-11-15 05:57:00+00",
    }

    assert polymarket_odds._market_dates(market) == {
        pd.Timestamp("2024-11-14").date(),
        pd.Timestamp("2024-11-15").date(),
    }


def test_the_more_traded_of_two_markets_for_one_game_wins(monkeypatch):
    """From March 2025 some games carry a near-empty `-moneyline` twin.

    Both are closed, both are typed moneyline, and only one has a usable price:
    nba-bos-mia-2025-03-14 traded $971,939 against the twin's $133.
    """
    real = {
        "slug": "nba-bos-mia-2025-03-14",
        "outcomes": '["Raptors","Bucks"]',
        "gameStartTime": "2025-03-14 23:00:00+00",
        "volume": "971939",
    }
    twin = {**real, "slug": "nba-bos-mia-2025-03-14-moneyline", "volume": "133"}

    for order in ([real, twin], [twin, real]):
        monkeypatch.setattr(polymarket_odds, "iter_nba_game_markets", lambda s, e, m=order: iter(m))
        index = polymarket_odds._index_markets(
            pd.Timestamp("2025-03-01").date(), pd.Timestamp("2025-04-01").date()
        )

        chosen = index[(frozenset({"Raptors", "Bucks"}), pd.Timestamp("2025-03-14").date())]
        assert chosen["slug"] == "nba-bos-mia-2025-03-14"


# ---------------------------------------------------------------------------
# Guarding the match rate
# ---------------------------------------------------------------------------


def _pending(season: str, count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"gameId": range(count), "season": [season] * count},
    )


def test_a_season_polymarket_never_listed_does_not_fail_the_build():
    """2026/27 in August has 55 NBA events and no games. That is the calendar."""
    pending = _pending("2026/27", 200)
    rows = [polymarket_odds._unresolved_row(game_id) for game_id in pending["gameId"]]

    polymarket_odds._check_match_rate(pending, rows)  # must not raise


def test_a_listed_season_that_mostly_misses_fails_loudly():
    """Half a season matching means a field we key on changed shape."""
    pending = _pending("2024/25", 200)
    rows = [polymarket_odds._unresolved_row(game_id) for game_id in pending["gameId"]]
    for row in rows[:100]:
        row["not_found"] = 0

    with pytest.raises(ValueError, match="below the"):
        polymarket_odds._check_match_rate(pending, rows)


def test_full_coverage_passes():
    pending = _pending("2024/25", 200)
    rows = [polymarket_odds._unresolved_row(game_id) for game_id in pending["gameId"]]
    for row in rows:
        row["not_found"] = 0

    polymarket_odds._check_match_rate(pending, rows)


# ---------------------------------------------------------------------------
# The market baseline
# ---------------------------------------------------------------------------


def _scored(game_ids):
    return pd.DataFrame(
        {
            "gameId": game_ids,
            "actual_home_win": [1] * len(game_ids),
            "prediction": [1] * len(game_ids),
            "home_win_probability": [0.8] * len(game_ids),
        }
    )


def test_a_game_missing_on_either_side_is_dropped_from_both():
    """The apples-to-apples guarantee: one n_games, not two."""
    scored = _scored([1, 2, 3])
    market = pd.DataFrame(
        {"gameId": [2, 3, 4], "market_home_win_probability": [0.6, 0.6, 0.6], "volume": [1e5] * 3}
    )

    comparison = polymarket_baseline.compare_to_market(scored, market)

    assert set(comparison["source"]) == {"model", "polymarket"}
    # Games 1 (no market) and 4 (no prediction) are gone from both rows.
    assert comparison["n_games"].tolist() == [2, 2]
    assert comparison["n_dropped"].tolist() == [1, 1]


def test_both_sides_are_scored_by_the_same_function():
    """Identical inputs must produce identical metrics, not merely similar ones."""
    scored = _scored([1, 2])
    market = pd.DataFrame(
        {"gameId": [1, 2], "market_home_win_probability": [0.8, 0.8], "volume": [1e5, 1e5]}
    )

    comparison = polymarket_baseline.compare_to_market(scored, market)

    model, polymarket = comparison.iloc[0], comparison.iloc[1]
    assert model["accuracy"] == polymarket["accuracy"]
    assert model["brier_score"] == pytest.approx(polymarket["brier_score"])


def test_no_mutual_coverage_returns_empty_rather_than_a_misleading_score():
    scored = _scored([1, 2])
    market = pd.DataFrame(
        {"gameId": [8, 9], "market_home_win_probability": [0.6, 0.6], "volume": [1e5, 1e5]}
    )

    assert polymarket_baseline.compare_to_market(scored, market).empty


def test_market_probabilities_are_normalised_to_remove_the_spread(tmp_path):
    """Both sides are independent quotes; the raw home price is biased high."""
    path = tmp_path / "odds.csv"
    pd.DataFrame(
        [
            {
                "gameId": 1, "game_slug": "s", "home_team": "gsw",
                "home_odds": 0.6, "home_qty": 1, "away_team": "lal", "away_odds": 0.5,
                "away_qty": 1, "not_found": 0, "price_source": "prices_history",
                "priced_at_utc": "", "volume": 1e5,
            }
        ]
    ).to_csv(path, index=False)

    market = polymarket_baseline.load_market_probabilities(path, min_volume=0)

    # 0.6 / (0.6 + 0.5), not 0.6.
    assert market["market_home_win_probability"].iloc[0] == pytest.approx(0.6 / 1.1)


def test_untraded_markets_are_excluded_by_volume(tmp_path):
    """2023/24's flat-0.5 markets are not opinions and must not enter the benchmark."""
    path = tmp_path / "odds.csv"
    pd.DataFrame(
        [
            {
                "gameId": 1, "game_slug": "s", "home_team": "gs", "home_odds": 0.5,
                "home_qty": 1, "away_team": "phx", "away_odds": 0.5, "away_qty": 1,
                "not_found": 0, "price_source": "prices_history", "priced_at_utc": "",
                "volume": 0.0,
            },
            {
                "gameId": 2, "game_slug": "t", "home_team": "gsw", "home_odds": 0.7,
                "home_qty": 1, "away_team": "lal", "away_odds": 0.3, "away_qty": 1,
                "not_found": 0, "price_source": "prices_history", "priced_at_utc": "",
                "volume": 1e6,
            },
        ]
    ).to_csv(path, index=False)

    market = polymarket_baseline.load_market_probabilities(path)

    assert market["gameId"].tolist() == [2]


def test_unresolved_rows_never_reach_the_benchmark(tmp_path):
    path = tmp_path / "odds.csv"
    pd.DataFrame([polymarket_odds._unresolved_row(1)]).to_csv(path, index=False)

    assert polymarket_baseline.load_market_probabilities(path, min_volume=0).empty


# ---------------------------------------------------------------------------
# Aligning the market onto test-split rows (what the comparison plots slice by)
# ---------------------------------------------------------------------------


def _market(game_ids, probabilities):
    return pd.DataFrame(
        {
            "gameId": game_ids,
            "market_home_win_probability": probabilities,
            "volume": [1e5] * len(game_ids),
        }
    )


def test_alignment_masks_the_rows_the_market_priced():
    game_ids = pd.Series([1, 2, 3, 4])
    market = _market([3, 1], [0.7, 0.3])

    mask, prices = polymarket_baseline.align_market_to_rows(game_ids, market)

    assert mask.tolist() == [True, False, True, False]
    # Prices come back in *row* order, not market order — the mask and the
    # prices have to index the same test arrays.
    assert prices.tolist() == [0.3, 0.7]


def test_alignment_survives_a_non_default_index():
    """`metadata_test` is a slice of the feature table, so its index is not 0..n."""
    game_ids = pd.Series([1, 2, 3], index=[104, 217, 980])
    market = _market([1, 3], [0.4, 0.9])

    mask, prices = polymarket_baseline.align_market_to_rows(game_ids, market)

    assert mask.tolist() == [True, False, True]
    assert prices.tolist() == [0.4, 0.9]


def test_alignment_collapses_a_duplicated_market_row():
    game_ids = pd.Series([1, 2])
    market = _market([1, 1, 2], [0.6, 0.6, 0.2])

    mask, prices = polymarket_baseline.align_market_to_rows(game_ids, market)

    assert mask.tolist() == [True, True]
    assert len(prices) == 2


def test_alignment_of_an_unpriced_split_is_empty_not_an_error():
    mask, prices = polymarket_baseline.align_market_to_rows(
        pd.Series([7, 8]), _market([1, 2], [0.6, 0.6])
    )

    assert not mask.any()
    assert len(prices) == 0


def test_alignment_without_an_odds_table_raises(monkeypatch):
    """The training run turns this into a skipped plot, not a failed run."""

    def _missing(*args, **kwargs):
        raise FileNotFoundError("No Polymarket odds at ...")

    monkeypatch.setattr(polymarket_baseline, "load_market_probabilities", _missing)

    with pytest.raises(FileNotFoundError):
        polymarket_baseline.align_market_to_rows(pd.Series([1, 2]))
