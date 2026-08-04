"""Canonical schema definitions for incremental games ingestion."""

CANONICAL_GAME_COLUMNS = [
    "gameId",
    "gameDate",
    "hometeamPrename",
    "hometeamName",
    "hometeamId",
    "awayteamPrename",
    "awayteamName",
    "awayteamId",
    "homeScore",
    "awayScore",
    "winner",
    "gameType",
    "attendance",
    "arenaId",
    "gameLabel",
    "gameSubLabel",
    "seriesGameNumber",
]

REQUIRED_GAME_COLUMNS = [
    "gameDate",
    "hometeamId",
    "awayteamId",
    "homeScore",
    "awayScore",
]

DEFAULT_GAME_VALUES = {
    "gameLabel": None,
    "gameSubLabel": None,
    "seriesGameNumber": None,
    "attendance": None,
    "arenaId": None,
}
