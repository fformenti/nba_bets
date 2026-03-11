"""
Project-wide constants for reusable defaults.
"""

# League schedule file
LEAGUE_SCHEDULE_FILE = "LeagueSchedule25_26.csv"

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


# Teams cities map
TEAMS_CITIES_MAP = {
    "Golden State": "San Francisco",
    "Indiana": "Indianapolis",
    "Utah": "Salt Lake City",
    "Washington": "Washington DC",
    "Minnesota": "Minneapolis",
}


# SERPER ENDPOINT
SERPER_ENDPOINT = "https://google.serper.dev/search"


list_of_nba_states = [
    "Arizona",
    "California",
    "Colorado",
    "Connecticut",
    "Florida",
    "Georgia",
    "Illinois",
    "Indiana",
    "Kansas",
    "Louisiana",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Missouri",
    "Nebraska",
    "Nevada",
    "New Jersey",
    "New York",
    "North Carolina",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Tennessee",
    "Texas",
    "Utah",
    "Washington",
    "Wisconsin",
    "District of Columbia",
    "Ontario",
    "British Columbia",
]

list_of_nba_cities = [
    "Philadelphia",
    "Chicago",
    "Memphis",
    "Atlanta",
    "Dallas",
    "Minneapolis",
    "Sacramento",
    "Houston",
    "Charlotte",
    "Newark",
    "Los Angeles",
    "Oklahoma City",
    "San Antonio",
    "Vancouver",
    "Miami",
    "San Francisco",
    "New York",
    "Portland",
    "Seattle",
    "Kansas City",
    "Cleveland",
    "Boston",
    "Orlando",
    "Salt Lake City",
    "Detroit",
    "Toronto",
    "New Orleans",
    "Milwaukee",
    "Denver",
    "Buffalo",
    "Washington",
    "Brooklyn",
    "San Diego",
    "Phoenix",
    "Indianapolis",
]
