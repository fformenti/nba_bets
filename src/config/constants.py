"""
Project-wide constants for reusable defaults.
"""

# Earliest date used for historical filtering (YYYY-MM-DD).
EARLIEST_GAME_DATE = "1980-08-01"

# Current season start year (e.g., 2025 for 2025/26 season).
CURRENT_SEASON_START_YEAR = 2025

# Neutral court game labels
INTERNATIONAL_GAMES = [
    "NBA Berlin Game",
    "NBA London Game",
    "NBA Mexico City Game",
    "NBA Paris Game",
    "NBA Paris Games",
]

LAS_VEGAS_GAMES = [
    "Emirates NBA Cup",
    "NBA Las Vegas Summer League",
]

# Combined list of all neutral court game labels
NEUTRAL_COURT_GAME_LABELS = INTERNATIONAL_GAMES + LAS_VEGAS_GAMES
