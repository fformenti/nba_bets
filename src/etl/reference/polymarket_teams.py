"""Hand-maintained NBA team -> Polymarket abbreviation mapping.

Polymarket identifies teams by a three-letter abbreviation that does not
always match ours, so the mapping is curated by hand and joined onto teamIds
here. Consumed by ``src/betting/slugs.py`` when building market slugs.

**One row per team, the current abbreviation.** These are ground truth from
Gamma's ``GET /teams?league=nba``, whose ``abbreviation`` field agrees with every
code below.

Five teams did use a different code once — Warriors ``gs``, Wizards ``wsh``,
Pelicans ``no``, Jazz ``utah`` (four characters), Spurs ``sa`` — but only through
the 2024 playoffs and the first week of 2024/25, and not once after 2024-10-28.
They are deliberately *not* carried here as variants. This table exists to build
a slug for an upcoming game, where a retired code cannot apply; the historical
backfill reads real slugs out of Gamma's listing instead of guessing them, so it
never needs a retired code either. A variant column was tried and deleted after
a full backfill matched 1,913 of 1,913 games without consulting it.

Phoenix is worth one note because it looks like drift and is not: ``phx``
resolves in every era and ``pho`` in none — that spelling comes from
Basketball-Reference.
"""

import pandas as pd

from src.config.paths import (
    TEAMS_CITIES_CONFERENCE_HISTORY_PROCESSED_PATH,
    POLYMARKET_TEAMS_ABV_PATH,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# The season every row is pinned to. This is the *merge key* onto the teams
# history table, not a claim about when an abbreviation was in use — franchises
# have neither moved nor been created since 2021, so one season's teamIds cover
# every historical game. Recency is expressed by `is_current`, not by this.
ABV_MERGE_SEASON = "2025/26"

nba_teams_abv = [
    {
        "city": "Atlanta",
        "team_name": "Hawks",
        "polymarket_abv": "ATL",
        "season": "2025/26",
    },
    {
        "city": "Boston",
        "team_name": "Celtics",
        "polymarket_abv": "BOS",
        "season": "2025/26",
    },
    {
        "city": "Brooklyn",
        "team_name": "Nets",
        "polymarket_abv": "BKN",
        "season": "2025/26",
    },
    {
        "city": "Charlotte",
        "team_name": "Hornets",
        "polymarket_abv": "CHA",
        "season": "2025/26",
    },
    {
        "city": "Chicago",
        "team_name": "Bulls",
        "polymarket_abv": "CHI",
        "season": "2025/26",
    },
    {
        "city": "Cleveland",
        "team_name": "Cavaliers",
        "polymarket_abv": "CLE",
        "season": "2025/26",
    },
    {
        "city": "Dallas",
        "team_name": "Mavericks",
        "polymarket_abv": "DAL",
        "season": "2025/26",
    },
    {
        "city": "Denver",
        "team_name": "Nuggets",
        "polymarket_abv": "DEN",
        "season": "2025/26",
    },
    {
        "city": "Detroit",
        "team_name": "Pistons",
        "polymarket_abv": "DET",
        "season": "2025/26",
    },
    {
        "city": "Golden State",
        "team_name": "Warriors",
        "polymarket_abv": "GSW",
        "season": "2025/26",
    },
    {
        "city": "Houston",
        "team_name": "Rockets",
        "polymarket_abv": "HOU",
        "season": "2025/26",
    },
    {
        "city": "Indiana",
        "team_name": "Pacers",
        "polymarket_abv": "IND",
        "season": "2025/26",
    },
    {
        "city": "Los Angeles",
        "team_name": "Clippers",
        "polymarket_abv": "LAC",
        "season": "2025/26",
    },
    {
        "city": "Los Angeles",
        "team_name": "Lakers",
        "polymarket_abv": "LAL",
        "season": "2025/26",
    },
    {
        "city": "Memphis",
        "team_name": "Grizzlies",
        "polymarket_abv": "MEM",
        "season": "2025/26",
    },
    {
        "city": "Miami",
        "team_name": "Heat",
        "polymarket_abv": "MIA",
        "season": "2025/26",
    },
    {
        "city": "Milwaukee",
        "team_name": "Bucks",
        "polymarket_abv": "MIL",
        "season": "2025/26",
    },
    {
        "city": "Minnesota",
        "team_name": "Timberwolves",
        "polymarket_abv": "MIN",
        "season": "2025/26",
    },
    {
        "city": "New Orleans",
        "team_name": "Pelicans",
        "polymarket_abv": "NOP",
        "season": "2025/26",
    },
    {
        "city": "New York",
        "team_name": "Knicks",
        "polymarket_abv": "NYK",
        "season": "2025/26",
    },
    {
        "city": "Oklahoma City",
        "team_name": "Thunder",
        "polymarket_abv": "OKC",
        "season": "2025/26",
    },
    {
        "city": "Orlando",
        "team_name": "Magic",
        "polymarket_abv": "ORL",
        "season": "2025/26",
    },
    {
        "city": "Philadelphia",
        "team_name": "76ers",
        "polymarket_abv": "PHI",
        "season": "2025/26",
    },
    {
        "city": "Phoenix",
        "team_name": "Suns",
        "polymarket_abv": "PHX",
        "season": "2025/26",
    },
    {
        "city": "Portland",
        "team_name": "Trail Blazers",
        "polymarket_abv": "POR",
        "season": "2025/26",
    },
    {
        "city": "Sacramento",
        "team_name": "Kings",
        "polymarket_abv": "SAC",
        "season": "2025/26",
    },
    {
        "city": "San Antonio",
        "team_name": "Spurs",
        "polymarket_abv": "SAS",
        "season": "2025/26",
    },
    {
        "city": "Toronto",
        "team_name": "Raptors",
        "polymarket_abv": "TOR",
        "season": "2025/26",
    },
    {
        "city": "Utah",
        "team_name": "Jazz",
        "polymarket_abv": "UTA",
        "season": "2025/26",
    },
    {
        "city": "Washington",
        "team_name": "Wizards",
        "polymarket_abv": "WAS",
        "season": "2025/26",
    },
]


OUTPUT_COLUMNS = ["teamId", "city", "team_name", "polymarket_abv", "season"]


def build_polymarket_teams_lookup() -> None:
    """Join the hand-maintained abbreviations onto teamIds and persist them.

    Previously this ran at import time, so merely importing the module read
    and rewrote a CSV.
    """
    teams_abv_df = pd.DataFrame(nba_teams_abv)
    teams_history_df = pd.read_csv(TEAMS_CITIES_CONFERENCE_HISTORY_PROCESSED_PATH)

    teams_abv_ids_df = teams_abv_df.merge(
        teams_history_df[["teamId", "teamCity", "teamName", "season"]],
        left_on=["city", "team_name", "season"],
        right_on=["teamCity", "teamName", "season"],
        how="left",
    )

    # A failed merge leaves teamId null, and `slugs.py` would then build slugs
    # for 29 teams and raise a KeyError on the 30th, mid-slate. Catch it here.
    unmatched = teams_abv_ids_df[teams_abv_ids_df["teamId"].isna()]
    if not unmatched.empty:
        offenders = sorted(unmatched["team_name"].unique().tolist())
        raise ValueError(
            f"{len(offenders)} team(s) have no teamId for season "
            f"'{ABV_MERGE_SEASON}' in {TEAMS_CITIES_CONFERENCE_HISTORY_PROCESSED_PATH}: "
            f"{offenders}. Run `make build-teams-history` first."
        )

    teams_abv_ids_df = teams_abv_ids_df[OUTPUT_COLUMNS]
    teams_abv_ids_df.to_csv(POLYMARKET_TEAMS_ABV_PATH, index=False)

    logger.info(
        f"Saved {len(teams_abv_ids_df)} team abbreviations to {POLYMARKET_TEAMS_ABV_PATH}"
    )
