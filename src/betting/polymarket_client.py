"""HTTP access to Polymarket's Gamma (events) and CLOB (price/book) APIs.

The only module that talks to Polymarket over the network. Everything else in
``src/betting`` works on the dataclasses returned here, so sizing and bet
construction stay testable without a live market.

**Two of Gamma's fields lie, and both cost a whole season if believed.**
``series_id`` is re-issued annually and the current one is not discoverable;
``sportsMarketType`` is wrong on dozens of real moneylines. This module filters
on the NBA *tag* and on the *slug shape* instead — see :data:`NBA_TAG_ID` and
:data:`GAME_SLUG`, which record what was measured rather than what is documented.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Iterator, Optional

import requests

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

# Polymarket occasionally stalls rather than refusing; without this a bet run
# can hang indefinitely mid-slate.
REQUEST_TIMEOUT_SECONDS = 15

# Gamma's NBA tag, and the only discovery key that survives a season rollover.
#
# `series_id` looks like the tidier filter and is a trap: Polymarket re-issues
# the series every year. Series 2 ("nba") stops dead at the 2025 Summer League,
# and 2025/26 lives in series 10345 ("nba-2026") — which `GET /series` does not
# list at all, so there is no way to discover the new id without already knowing
# a game slug. A series-keyed backfill returns *zero* games every October and
# says nothing about it.
#
# The tag does return futures and player props alongside games, which is the
# reason it was passed over before. That is handled structurally by GAME_SLUG
# below, in the same pass, at a cost of roughly 25% more bytes on a discovery
# step that takes under two minutes.
NBA_TAG_ID = 745

# A game moneyline's slug, and nothing else.
#
# `sportsMarketType` cannot do this job. It is wrong on 49 markets in 2024/25 —
# 34 of them Thunder away games — which carry two team nicknames, real volume,
# and the type "totals": `nba-okc-den-2024-10-24` is the only market on its
# event and is a moneyline in everything but the label.
#
# The slug is trustworthy where the type is not. Spread and total markets always
# carry a suffix (`-spread-away-3pt5`, `-total-over-220pt5`), so anchoring the
# pattern is what excludes them. The optional `-moneyline` suffix is a real
# variant that appeared in March 2025.
GAME_SLUG = re.compile(
    r"^nba-(?P<first>[a-z]{2,4})-(?P<second>[a-z]{2,4})-(?P<date>\d{4}-\d{2}-\d{2})(-moneyline)?$"
)

# Gamma caps a page at 100 and *silently* ignores anything larger — limit=1000
# returns a byte-identical 100 rows.
GAMMA_PAGE_LIMIT = 100

# `/events` refuses `offset` past 2000 with a 422 whose message names its own
# replacement: "offset too large, use /events/keyset for deeper pagination".
#
# That advice does not work. `/events/keyset` accepts the date filters and then
# ignores its own cursor: page 2 onwards returns a `next_cursor` identical to
# the one sent, and the same 100 events come back forever (2,500 rows, 100
# distinct, when measured). It cannot page a filtered query at all.
#
# So the window is split into one-year chunks instead, each of which fits inside
# the offset ceiling with room to spare — the busiest NBA year on record is 1,776
# events against a 2,000 cap.
GAMMA_MAX_OFFSET = 2000

# Widest window CLOB's /prices-history is asked for around a tip-off. The
# endpoint truncates at roughly 3000 points without reporting it, so at
# fidelity=1 anything past ~2 days would be silently cut short. Six hours is
# also more than enough: across 1,913 backfilled games every one resolved inside
# this window, with a median of one minute between the last price and tip-off.
PRICE_LOOKBACK = timedelta(hours=6)

# Pricing a season is a few thousand small requests. Eight in flight measured at
# 120 requests in 7.7s with no 429s and no throttling, against ~40 minutes
# serial. Kept modest deliberately: Polymarket publishes no rate limit, so this
# is a number that was observed to be safe rather than one that was permitted.
MAX_WORKERS = 8

# CLOB times out occasionally under sustained load — four times across a 3,826
# request backfill, all of which succeeded on the next attempt. Without retries
# a full run aborts partway through on a fault that is not really an error.
REQUEST_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.5


@lru_cache(maxsize=1)
def _session() -> requests.Session:
    """Shared HTTP session, built on first use.

    A backfill makes a few thousand requests to two hosts; connection reuse is
    most of the wall time. Lazily constructed rather than module-level so
    importing this module still opens no sockets.
    """
    session = requests.Session()
    # One connection per worker, or the pool serialises what the threads set out
    # to parallelise (and urllib3 warns on every discarded connection).
    adapter = requests.adapters.HTTPAdapter(pool_maxsize=MAX_WORKERS * 2)
    session.mount("https://", adapter)
    return session


def _get(url: str, params: Optional[dict] = None) -> requests.Response:
    """GET with a bounded retry, for transport faults only.

    Retries connection errors and timeouts, which Polymarket produces sporadically
    under load. HTTP status codes are *not* retried: a 404 is an answer, and a
    500 that repeats is worth surfacing rather than hiding behind three more
    attempts.
    """
    for attempt in range(REQUEST_RETRIES + 1):
        try:
            return _session().get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException as error:
            if attempt == REQUEST_RETRIES:
                raise
            logger.debug(f"{type(error).__name__} on {url}; retry {attempt + 1}")
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise AssertionError("unreachable")  # pragma: no cover


@lru_cache(maxsize=1)
def nba_team_nicknames() -> frozenset:
    """Every team nickname Polymarket uses in a market's ``outcomes``.

    Markets name their sides by nickname — ``["Thunder", "Nuggets"]`` — in every
    era, which is what makes nickname matching immune to the abbreviation drift
    and slug-order changes that the *slugs* went through. This is the vocabulary
    that separates a game market from a prop.

    ``GET /teams?league=nba`` returns 37 rows: the 30 franchises plus 7 All-Star
    squads (``Shaq's OGs`` and friends). The All-Star names are harmless here —
    no real game will ever be keyed on them.
    """
    response = _get(f"{GAMMA_API}/teams", params={"league": "nba", "limit": 100})
    response.raise_for_status()
    return frozenset(
        str(team["name"]).strip() for team in response.json() or [] if team.get("name")
    )


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

    That ordering holds from 2024/25 on, which is every game this is used for —
    it serves the live betting path. It is **inverted for 2023/24**, where slugs
    and outcomes are home-team-first, so do not reuse this for a historical
    backfill. :mod:`src.betting.polymarket_odds` takes orientation from our own
    hometeamId instead of trusting the market's ordering.
    """
    response = _get(f"{GAMMA_API}/events/slug/{game_slug}")
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
    """Fetch the CLOB order book for one outcome token.

    Live markets only. A closed market has no book: CLOB answers 404 with
    "No orderbook exists for the requested token id". Historical prices come
    from :func:`get_price_at` instead.
    """
    response = _get(f"{CLOB_API}/book", params={"token_id": token_id})
    response.raise_for_status()
    return response.json()


def get_settled_outcomes(game_slug: str) -> Optional[dict]:
    """Settlement result per outcome for a game, as ``{team_name: 0.0 | 1.0}``.

    **This is not a price.** Gamma's ``outcomePrices`` on a settled market is the
    resolution label — ``["1","0"]`` for a home win, ``["0","1"]`` for an away
    win — so it says who won, at no point what the market thought beforehand.
    An earlier version of this function was named ``get_market_prices`` and
    documented as returning closing prices; it would have made the model look
    like it was being benchmarked against an oracle.

    For what the market actually believed at tip-off, use :func:`get_price_at`.
    Returns ``None`` when the market is unknown or unsettled.
    """
    market = get_market_by_slug(game_slug)
    if market is None or "outcomePrices" not in market:
        return None

    team_names = _as_list(market["outcomes"])
    prices = [float(p) for p in _as_list(market["outcomePrices"])]
    return dict(zip(team_names, prices, strict=True))


def _as_list(value) -> list:
    """Gamma returns list-valued fields as JSON strings on some endpoints."""
    return json.loads(value) if isinstance(value, str) else value


def is_game_market(market: dict) -> bool:
    """True for a two-team moneyline, excluding spreads, totals, futures and props.

    Two conditions, neither of which is ``sportsMarketType`` — see
    :data:`GAME_SLUG` for why that field cannot be trusted:

    1. the slug is a bare ``nba-{abv}-{abv}-{date}``, which every spread and
       total fails because they append their line to it;
    2. both ``outcomes`` are NBA team nicknames, which excludes the futures that
       share the slug shape by coincidence and every Yes/No prop.

    Measured over the 2024/25 tag listing, this keeps 1,531 game moneylines and
    rejects all 2,070 spreads.
    """
    if not GAME_SLUG.match(market.get("slug") or ""):
        return False

    outcomes = [str(name).strip() for name in _as_list(market.get("outcomes") or [])]
    if len(outcomes) != 2:
        return False

    nicknames = nba_team_nicknames()
    return all(name in nicknames for name in outcomes)


def iter_nba_game_markets(start: date, end: date) -> Iterator[dict]:
    """Every NBA game moneyline whose game falls in ``[start, end)``.

    This is how a backfill finds markets. Constructing a slug per game and asking
    for it costs ~1,850 requests a season, and it guesses at exactly the things
    that turn out to vary — abbreviations, team order, an optional suffix. Paging
    the event listing costs 18 requests for a season and returns the true slug
    rather than a plausible one.

    Filtering is on ``end_date``, which tracks the game itself. ``start_date`` is
    market *creation* time and carries placeholder values on legacy events, so
    filtering by it silently drops whole seasons.

    The window is walked one year at a time — see :data:`GAMMA_MAX_OFFSET` for
    why paging it in one span is not possible.

    Yields raw market dicts — ``slug``, ``conditionId``, ``clobTokenIds``,
    ``gameStartTime``, ``outcomes``, ``volume`` — flattened out of their events.
    """
    for chunk_start, chunk_end in _yearly_windows(start, end):
        yield from _iter_markets_in_window(chunk_start, chunk_end)


def _yearly_windows(start: date, end: date) -> Iterator[tuple[date, date]]:
    """``[start, end)`` split on anniversaries of ``start``."""
    window_start = start
    while window_start < end:
        window_end = min(window_start.replace(year=window_start.year + 1), end)
        yield window_start, window_end
        window_start = window_end


def _event_count(start: date, end: date) -> Optional[int]:
    """How many events Gamma says are in the window, or ``None`` if it will not say."""
    response = _get(
        f"{GAMMA_API}/events/pagination",
        params={
            "tag_id": NBA_TAG_ID,
            "end_date_min": f"{start.isoformat()}T00:00:00Z",
            "end_date_max": f"{end.isoformat()}T00:00:00Z",
            "limit": 1,
        },
    )
    if not response.ok:
        return None
    return (response.json() or {}).get("pagination", {}).get("totalResults")


def _iter_markets_in_window(start: date, end: date) -> Iterator[dict]:
    """Game markets in one window small enough to offset-page.

    Ordered by ``id``, which is unique. ``endDate`` seems the natural sort and
    quietly loses games: dozens of events share a tip-off minute, the backend's
    ordering among them is not stable across requests, and a row that moves
    between two page reads is never returned. Over the 2024/25 window that cost
    one event in 1,776 — a real game with $536,588 of volume, indistinguishable
    afterwards from a game Polymarket never listed.

    The expected count is checked afterwards so the same class of bug cannot
    recur silently.
    """
    expected = _event_count(start, end)
    seen = set()
    offset = 0

    while True:
        response = _get(
            f"{GAMMA_API}/events",
            params={
                "tag_id": NBA_TAG_ID,
                "end_date_min": f"{start.isoformat()}T00:00:00Z",
                "end_date_max": f"{end.isoformat()}T00:00:00Z",
                "order": "id",
                "ascending": "true",
                "limit": GAMMA_PAGE_LIMIT,
                "offset": offset,
            },
        )
        response.raise_for_status()
        events = response.json() or []
        if not events:
            break

        for event in events:
            seen.add(event.get("id"))
            for market in event.get("markets") or []:
                if is_game_market(market):
                    yield market

        if len(events) < GAMMA_PAGE_LIMIT:
            break

        offset += GAMMA_PAGE_LIMIT
        if offset >= GAMMA_MAX_OFFSET:
            # A year of NBA events outgrew the offset ceiling. Silently returning
            # a truncated season would look exactly like Polymarket not listing
            # those games, so say it instead.
            raise RuntimeError(
                f"More than {GAMMA_MAX_OFFSET} NBA events between {start} and {end}; "
                "Gamma cannot page past that. Split the window more finely than a year."
            )

    if expected is not None and len(seen) < expected:
        logger.warning(
            f"Gamma listed {expected} event(s) between {start} and {end} but paging "
            f"returned {len(seen)}; {expected - len(seen)} may be missing from the odds table."
        )


def get_market_by_slug(game_slug: str) -> Optional[dict]:
    """One market by its slug, or ``None`` if Polymarket has no such market.

    Uses ``/markets/slug/`` rather than ``/events/slug/`` because only the
    former works in both eras — a 2023/24 game is a market nested in a
    ``nba-dailies-{date}`` event, so the event lookup 404s on every game that
    season.
    """
    response = _get(f"{GAMMA_API}/markets/slug/{game_slug}")
    if response.status_code == 404:
        return None
    response.raise_for_status()

    payload = response.json()
    # The endpoint has returned both a bare market and a single-element list.
    if isinstance(payload, list):
        payload = payload[0] if payload else None
    return payload or None


def get_price_at(token_id: str, when: datetime) -> Optional[tuple[float, datetime]]:
    """Last traded price at or before ``when``, with the time it was quoted.

    Returns ``(price, quoted_at)``, or ``None`` if the market never traded in
    the lookback window.

    Uses explicit ``startTs``/``endTs``: the ``interval`` parameter returns an
    empty history for closed markets, which is every market a backfill touches.
    The window is deliberately short — the endpoint truncates at roughly 3000
    points and reports no error when it does.

    A single window is enough. Widening it on a miss was tried and never once
    helped: of 1,913 backfilled games, all 1,913 resolved on the first attempt.
    """
    end_ts = int(when.timestamp())
    start_ts = int((when - PRICE_LOOKBACK).timestamp())

    response = _get(
        f"{CLOB_API}/prices-history",
        params={"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": 1},
    )
    response.raise_for_status()

    history = response.json().get("history") or []
    points = [point for point in history if point["t"] <= end_ts]
    if not points:
        return None

    latest = max(points, key=lambda point: point["t"])
    return float(latest["p"]), datetime.fromtimestamp(latest["t"], tz=timezone.utc)


# There is deliberately no /trades fallback here. Data-API's public trade feed
# pages newest-first and caps `offset` at 10,000 with no timestamp filter, so on
# a liquid game the pre-tip-off trades are unreachable — it fails exactly where
# the price matters most. It was carried as a fallback until a full backfill
# showed get_price_at answering for every one of 1,913 games, at which point an
# untested path that cannot serve its own purpose was worth deleting.
