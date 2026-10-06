"""Backfill Polymarket's tip-off price for every game we have a prediction for.

The benchmark the model is measured against. Accuracy against outcomes says
nothing on its own; accuracy against what a liquid market believed at tip-off
says whether the model is worth running.

**What is and is not retrievable.** Gamma's ``outcomePrices`` on a settled
market is the resolution label (``["1","0"]``), not a closing price — reading it
as odds would benchmark the model against an oracle. The real pre-game price
comes from CLOB's ``/prices-history``, which does retain data for markets closed
years ago. There is no order book for a closed market, so depth cannot be
recovered at all; ``home_qty``/``away_qty`` carry a placeholder ``1`` to mark
"priced, but no book" rather than implying a size that was never observed.

**How games are matched.** Not by building a slug and hoping. Markets are listed
by season and matched to our games on ``{home team name, away team name}`` plus a
date, because the market names its sides by full nickname in every era while its
*slug* keeps changing shape — 2023/24 puts the home team first and nests games
inside a daily event, 2024/25 puts the away team first with one event per game,
and March 2025 added an optional ``-moneyline`` suffix. Matching on names and
taking orientation from our own ``hometeamId`` makes all of that irrelevant.

Run via ``make build-polymarket-odds``.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from typing import Iterable, Optional

import pandas as pd

from src.betting.game_slugs import load_games_since
from src.betting.polymarket_client import (
    GAME_SLUG,
    MAX_WORKERS,
    get_price_at,
    iter_nba_game_markets,
)
from src.config.constants import SCHEDULE_TIMEZONE
from src.config.paths import INGESTED_DIR, INGESTED_POLYMARKET_ODDS_PATH
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# First season Polymarket priced NBA regular-season games at all.
#
# 2023/24 looks available and is not: its only NBA game markets run from
# 2024-04-20, the playoffs. Zero of that season's 1,230 regular-season games
# have a market, so including it would add a season of `not_found = 1` rows and
# a discovery page fetched for nothing.
#
# It coincides with `splitting.test_start_season`, which is a convenience rather
# than a dependency — the benchmark would be worth having over more history if
# Polymarket had priced more history.
DEFAULT_START_SEASON = "2024/25"

COLUMNS = [
    "gameId",
    "game_slug",
    "home_team",
    "home_odds",
    "home_qty",
    "away_team",
    "away_odds",
    "away_qty",
    "not_found",
    "price_source",
    "priced_at_utc",
    "volume",
]

# No order book survives a market's close, so a historical row records depth it
# never observed. The 1 marks "this side was priced"; 0 marks "not found".
PLACEHOLDER_QTY = 1

# Below this share of a season's games matched, something structural has broken —
# a re-issued tag, a changed outcome vocabulary, a new slug shape — and the run
# would otherwise write a table of not_found=1 that reads as "Polymarket had no
# markets". Real coverage on a listed season is 100%, so this is a wide margin,
# not a target. It is judged per season and only on seasons where at least one
# market was found: a season Polymarket has not begun listing is a fact about
# the calendar, not a failure.
MIN_MATCH_RATE = 0.9


def _eastern_date(timestamp: str) -> Optional[date]:
    """Calendar date of a UTC tip-off, in the league's own timezone.

    Polymarket stamps ``gameStartTime`` in UTC, which rolls past midnight for
    every evening game — a 7pm Eastern tip-off on the 24th is 00:00 UTC on the
    25th. Converting through ``SCHEDULE_TIMEZONE`` is what makes it agree with
    our ``gameDateOnlyStr``.
    """
    parsed = pd.to_datetime(timestamp, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.tz_convert(SCHEDULE_TIMEZONE).date()


def _as_list(value) -> list:
    return json.loads(value) if isinstance(value, str) else (value or [])


def _matchup(market: dict) -> Optional[frozenset]:
    """The two team nicknames a market is between."""
    outcomes = _as_list(market.get("outcomes"))
    if len(outcomes) != 2:
        return None
    return frozenset(str(name).strip() for name in outcomes)


def _market_dates(market: dict) -> set:
    """Every calendar date a market could reasonably be filed under.

    Two, because Polymarket's own two date fields disagree for late tip-offs.
    ``nba-dal-uta-2024-11-14`` starts at 2024-11-15 05:57 UTC — 00:57 Eastern,
    so its Eastern calendar date is the 15th while the game, its slug, and our
    ``gameDateOnlyStr`` all say the 14th. Filing the market under both dates
    matches every game exactly; keying on the tip-off alone missed 89 of 1,913
    and needed a ±1-day fudge that risks mismatching a back-to-back.
    """
    dates = set()

    slug_match = GAME_SLUG.match(market.get("slug") or "")
    if slug_match:
        dates.add(date.fromisoformat(slug_match.group("date")))

    tipoff_date = _eastern_date(market.get("gameStartTime") or market.get("endDate"))
    if tipoff_date is not None:
        dates.add(tipoff_date)

    return dates


def _game_matchup(row) -> frozenset:
    return frozenset({str(row.hometeamName).strip(), str(row.awayteamName).strip()})


def _season_span(seasons: Iterable[str]) -> tuple[date, date]:
    """Date window covering every season in ``seasons``.

    A season labelled ``2024/25`` runs from autumn 2024 into summer 2025, so the
    window opens in the July before the first season and closes in the August
    after the last one ends — a year and a bit after that season's start year.
    """
    start_years = sorted(int(season.split("/")[0]) for season in seasons)
    return date(start_years[0], 7, 1), date(start_years[-1] + 1, 8, 1)


def _index_markets(start: date, end: date) -> dict:
    """Every NBA game market in the window, keyed by ``(matchup, date)``.

    Duplicates are resolved by volume, which is load-bearing rather than
    defensive. From March 2025 many games carry two markets — the real one and a
    near-empty twin with a ``-moneyline`` slug suffix. ``nba-bos-mia-2025-03-14``
    traded $971,939 while ``nba-bos-mia-2025-03-14-moneyline`` traded $133; both
    are live, closed, correctly typed, and only one has a usable price. 200 of
    1,913 games are affected.
    """
    index: dict = {}
    duplicates = 0
    for market in iter_nba_game_markets(start, end):
        matchup = _matchup(market)
        if matchup is None:
            continue

        for game_date in _market_dates(market):
            key = (matchup, game_date)
            if key in index:
                duplicates += 1
                if _volume(market) <= _volume(index[key]):
                    continue
            index[key] = market

    logger.info(
        f"Indexed {len(index)} Polymarket game market key(s) between {start} and {end}"
        + (f" ({duplicates} duplicate(s) resolved by volume)" if duplicates else "")
    )
    return index


def _volume(market: dict) -> float:
    try:
        return float(market.get("volume") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _tipoff(market: dict, row) -> datetime:
    """Tip-off as a UTC instant, preferring Polymarket's own stamp.

    Our ``gameDate`` is Eastern wall time carried without an offset, so it has
    to be localised rather than read as UTC.
    """
    stamped = pd.to_datetime(market.get("gameStartTime"), utc=True, errors="coerce")
    if not pd.isna(stamped):
        return stamped.to_pydatetime()

    local = pd.to_datetime(row.gameDate).tz_localize(SCHEDULE_TIMEZONE)
    return local.tz_convert("UTC").to_pydatetime()


def _side_price(
    token_ids: list, outcome_index: int, when: datetime
) -> tuple[Optional[float], Optional[datetime], str]:
    """Price for one side at tip-off, and where it came from."""
    quoted = get_price_at(token_ids[outcome_index], when)
    if quoted is None:
        return None, None, "none"
    return quoted[0], quoted[1], "prices_history"


def _unresolved_row(game_id) -> dict:
    return {
        "gameId": game_id,
        "game_slug": "",
        "home_team": "",
        "home_odds": 0.0,
        "home_qty": 0,
        "away_team": "",
        "away_odds": 0.0,
        "away_qty": 0,
        "not_found": 1,
        "price_source": "none",
        "priced_at_utc": "",
        "volume": 0.0,
    }


def _abbreviations(slug: str) -> tuple[str, str]:
    """The two team codes in a slug, in the order the slug lists them.

    Parsed rather than looked up, so the row records the code Polymarket
    actually used. That matters because the codes drifted — ``gs``/``gsw``,
    ``no``/``nop``, ``sa``/``sas``, ``utah``/``uta``, ``wsh``/``was`` — through
    the 2024 playoffs and the first week of 2024/25, settling on the three-letter
    forms after 2024-10-28.
    """
    match = GAME_SLUG.match(slug or "")
    if match is None:
        return "", ""
    return match.group("first"), match.group("second")


def _resolve_market(row, index: dict) -> Optional[dict]:
    """The Polymarket market for one game, or ``None`` if it listed none.

    A single dictionary lookup. There is deliberately no slug-guessing fallback:
    the listing is authoritative, and over 1,913 games across two seasons it
    matched every single one, leaving a fallback that could only ever fire when
    the answer was genuinely "no market exists".
    """
    return index.get((_game_matchup(row), pd.to_datetime(row.gameDateOnlyStr).date()))


def _build_row(row, market: dict) -> dict:
    """One output row from a game and the market it matched."""
    outcomes = [str(name).strip() for name in _as_list(market.get("outcomes"))]
    token_ids = _as_list(market.get("clobTokenIds"))
    if len(outcomes) != 2 or len(token_ids) != 2:
        return _unresolved_row(row.gameId)

    # Orientation comes from our own record, never from the market's ordering —
    # which is away-first from 2024/25 but home-first across 2023/24.
    home_name = str(row.hometeamName).strip()
    if home_name not in outcomes:
        return _unresolved_row(row.gameId)

    home_index = outcomes.index(home_name)
    away_index = 1 - home_index

    slug = market.get("slug", "")
    first_abv, second_abv = _abbreviations(slug)
    home_abv = first_abv if home_index == 0 else second_abv
    away_abv = second_abv if home_index == 0 else first_abv

    when = _tipoff(market, row)
    home_odds, home_at, home_source = _side_price(token_ids, home_index, when)
    away_odds, away_at, away_source = _side_price(token_ids, away_index, when)

    if home_odds is None and away_odds is None:
        # The market exists but never traded before tip-off. That is a real
        # finding about the market, not a missing game, so the slug and volume
        # are kept while the odds stay zeroed.
        unresolved = _unresolved_row(row.gameId)
        unresolved.update(
            {"game_slug": slug, "home_team": home_abv, "away_team": away_abv,
             "volume": _volume(market)}
        )
        return unresolved

    # One side quoting and the other not is normal on a thin market; the missing
    # side is implied by its complement.
    if home_odds is None:
        home_odds, home_at, home_source = 1.0 - away_odds, away_at, f"{away_source}_implied"
    if away_odds is None:
        away_odds, away_at, away_source = 1.0 - home_odds, home_at, f"{home_source}_implied"

    priced_at = max(stamp for stamp in (home_at, away_at) if stamp is not None)
    source = home_source if home_source == away_source else f"{home_source}+{away_source}"

    return {
        "gameId": row.gameId,
        "game_slug": slug,
        "home_team": home_abv,
        "home_odds": round(home_odds, 4),
        "home_qty": PLACEHOLDER_QTY,
        "away_team": away_abv,
        "away_odds": round(away_odds, 4),
        "away_qty": PLACEHOLDER_QTY,
        "not_found": 0,
        "price_source": source,
        "priced_at_utc": priced_at.isoformat(),
        "volume": _volume(market),
    }


def _load_existing() -> pd.DataFrame:
    if not INGESTED_POLYMARKET_ODDS_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(INGESTED_POLYMARKET_ODDS_PATH)


def build_polymarket_odds(
    start_season: str = DEFAULT_START_SEASON,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch and persist Polymarket's tip-off odds for every game since ``start_season``.

    Re-running is cheap and safe: games already in the table are skipped, so the
    same command backfills three past seasons on its first run and keeps the
    current season current on every run after. ``refresh`` additionally retries
    the rows that previously resolved to nothing — worth doing once a season has
    finished and its markets have settled, pointless mid-season.
    """
    games = load_games_since(start_season)
    existing = _load_existing()

    known = set(existing["gameId"]) if not existing.empty else set()
    if refresh and not existing.empty:
        known -= set(existing.loc[existing["not_found"] == 1, "gameId"])
        existing = existing[existing["not_found"] != 1]

    pending = games[~games["gameId"].isin(known)]
    if pending.empty:
        logger.info(f"All {len(games)} game(s) already priced in {INGESTED_POLYMARKET_ODDS_PATH}")
        return existing

    start, end = _season_span(pending["season"].unique())
    logger.info(
        f"Pricing {len(pending)} game(s) from season {start_season} on "
        f"({len(known)} already known)"
    )
    index = _index_markets(start, end)

    resolved = [(row, _resolve_market(row, index)) for row in pending.itertuples()]

    def build(pair) -> dict:
        row, market = pair
        return _unresolved_row(row.gameId) if market is None else _build_row(row, market)

    # Each game is two independent price lookups and nothing else, so the whole
    # loop is network latency. Threading it takes a full backfill from ~40
    # minutes to ~5; see MAX_WORKERS for why the number is what it is.
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        rows = list(pool.map(build, resolved))

    matched = sum(row["not_found"] == 0 for row in rows)
    _check_match_rate(pending, rows)

    fetched = pd.DataFrame(rows, columns=COLUMNS)
    # Concatenating onto an all-NA empty frame would let pandas infer column
    # dtypes from nothing, turning not_found into a float.
    combined = fetched if existing.empty else pd.concat([existing, fetched], ignore_index=True)
    combined = combined.drop_duplicates(subset=["gameId"], keep="last").sort_values("gameId")

    INGESTED_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(INGESTED_POLYMARKET_ODDS_PATH, index=False)

    logger.info(
        f"Priced {matched}/{len(pending)} game(s); "
        f"{len(combined)} row(s) total → {INGESTED_POLYMARKET_ODDS_PATH}"
    )
    _log_coverage(games, combined)
    return combined


def _check_match_rate(pending: pd.DataFrame, rows: list) -> None:
    """Fail the run if a season Polymarket *did* list came back mostly empty.

    Judged per season, and skipped entirely for a season with no markets at all.
    Those are two different situations that a single overall rate would blur: a
    season Polymarket has not started listing (2026/27, in August) is a fact
    about the calendar, while a listed season that suddenly matches 40% means a
    field this module keys on has changed shape underneath it.
    """
    resolved_by_game = {row["gameId"]: row["not_found"] == 0 for row in rows}
    seasons = pending.assign(
        resolved=pending["gameId"].map(resolved_by_game).fillna(False)
    ).groupby("season")["resolved"]

    for season, group in seasons:
        matched, total = int(group.sum()), len(group)
        if matched == 0:
            logger.info(f"  {season}: no Polymarket markets listed — nothing to price")
            continue

        rate = matched / total
        if rate < MIN_MATCH_RATE:
            raise ValueError(
                f"Season {season}: only {matched}/{total} games ({rate:.1%}) matched a "
                f"Polymarket market, below the {MIN_MATCH_RATE:.0%} floor. Polymarket "
                "listed markets that season, so this is a broken matcher rather than "
                "missing coverage — check that team nicknames still agree with the "
                "`outcomes` field and that game slugs still match GAME_SLUG."
            )


def _log_coverage(games: pd.DataFrame, odds: pd.DataFrame) -> None:
    """Per-season resolved counts, so a bad season is visible without a query."""
    merged = games[["gameId", "season"]].merge(odds[["gameId", "not_found"]], on="gameId")
    for season, group in merged.groupby("season"):
        resolved = int((group["not_found"] == 0).sum())
        logger.info(f"  {season}: {resolved}/{len(group)} priced")
