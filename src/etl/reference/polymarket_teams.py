"""Hand-maintained NBA team -> Polymarket abbreviation mapping.

Polymarket identifies teams by a three-letter abbreviation that does not
always match ours, so the mapping is curated by hand and joined onto teamIds
here. Consumed by ``src/betting/slugs.py`` when building market slugs.
"""

import pandas as pd

from src.config.paths import (
    TEAMS_CITIES_CONFERENCE_HISTORY_PROCESSED_PATH,
    POLYMARKET_TEAMS_ABV_PATH,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

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


def build_polymarket_teams_lookup() -> None:
    """Join the hand-maintained abbreviations onto teamIds and persist them.

    Previously this ran at import time, so merely importing the module read
    and rewrote a CSV.
    """
    teams_abv_df = pd.DataFrame(
        nba_teams_abv, columns=["city", "team_name", "polymarket_abv", "season"]
    )
    teams_history_df = pd.read_csv(TEAMS_CITIES_CONFERENCE_HISTORY_PROCESSED_PATH)

    teams_abv_ids_df = teams_abv_df.merge(
        teams_history_df[["teamId", "teamCity", "teamName", "season"]],
        left_on=["city", "team_name", "season"],
        right_on=["teamCity", "teamName", "season"],
        how="left",
    )

    teams_abv_ids_df = teams_abv_ids_df[["teamId", "city", "team_name", "polymarket_abv", "season"]]
    teams_abv_ids_df.to_csv(POLYMARKET_TEAMS_ABV_PATH, index=False)
    logger.info(
        f"Saved {len(teams_abv_ids_df)} team abbreviations to {POLYMARKET_TEAMS_ABV_PATH}"
    )
