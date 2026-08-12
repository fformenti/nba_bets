"""
Project-wide constants for reusable defaults.
"""

# League schedule file
LEAGUE_SCHEDULE_FILE = "LeagueSchedule25_26.csv"

# Current season start year (e.g., 2025 for 2025/26 season).
CURRENT_SEASON_START_YEAR = 2025

# Earliest season kept in processed data. Seasons before this are dropped:
# the teams history lookup (src/etl/reference/teams_history.py) has sparse
# conference coverage before 1950/51, which would otherwise leave games with
# NaN conference features.
MIN_SEASON = "1950/51"

# ===== Incremental collection =====
# How many times a pending game may come back without a terminal status *after
# its scheduled tip-off* before it is quarantined. Without this the daily cycle
# deadlocks: the selector keeps emitting the same unanswerable game and the
# history frontier never advances.
MAX_FETCH_ATTEMPTS = 5

# The clock the league schedule's tip-off times are expressed in. The raw
# gameDateTimeEst column stamps a "Z" on values that are really Eastern wall
# time, and the collectors carry them naive rather than re-labelling them, so
# anything comparing a payload's gameDate against "now" must use this zone.
SCHEDULE_TIMEZONE = "America/New_York"

# Bookkeeping the results collector owns, carried on a pending payload so a game
# remembers how many times it has been asked about across runs. It lives here
# because the selector has to preserve it when refreshing a payload.
FETCH_ATTEMPTS_KEY = "_fetchAttempts"

# A scheduled game this many days behind the history frontier is quarantined
# even if it was never fetched. Second line of defence, for games that fall out
# of the pending directory entirely.
UNRESOLVED_GRACE_DAYS = 3

# The third digit of a zero-padded NBA gameId encodes what kind of game it is.
# This is the only reliable discriminator for incrementally collected games: the
# league schedule's gameLabel does not mark playoff rounds, and results payloads
# carry no gameType at all.
GAME_ID_TYPE_PREFIX = {
    "1": "Preseason",
    "2": "Regular Season",
    "3": "All-Star",
    "4": "Playoffs",
    "5": "Play-in Tournament",
}

# Only these game types belong in the regular-season table.
REGULAR_SEASON_GAME_TYPE = "Regular Season"

ENRICHED_COLUMNS = [
    "win_bool",
    "pts_diff",
    "games_played_HT",
    "games_played_VT",
    "total_wins_HT",
    "total_losses_HT",
    "total_wins_VT",
    "total_losses_VT",
]

# Intermediate columns produced by ETL that should not be used as model features.
# Delta-input columns (HT/VT pairs) are handled by create_delta_features().
# days_at_home/days_on_road are handled by the home_and_road feature group.
INTERMEDIATE_COLUMNS = [
    "pts_diff",
    "distance",
    "total_wins_HT",
    "total_losses_HT",
    "total_wins_VT",
    "total_losses_VT",
    "total_wins_HT_at_home",
    "total_losses_HT_at_home",
    "total_wins_VT_on_road",
    "total_losses_VT_on_road",
    "games_played_HT_at_home",
    "games_played_VT_on_road",
    # "conf_rank_HT",
    # "conf_rank_VT",
]

DEFAULT_METADATA_COLUMNS = [
    "gameId",
    "winner",
    "hometeamPrename",
    "hometeamId",
    "hometeamName",
    "homeScore",
    "awayteamName",
    "awayteamPrename",
    "awayteamId",
    "awayScore",
    "hometeamLocation",
    "awayteamLocation",
    "gameLocation",
    "hometeamConference",
    "awayteamConference",
    "winnerteamConference",
    "gameType",
    "gameDateOnlyStr",
    "season",
    "overtimes",
    "postponed",
    "is_neutral_court_game",
]



# Precision for floats written to feature CSVs.  Feature math runs at full
# double precision; rounding happens only here, at the persistence boundary,
# so intermediate results (e.g. SOS feeding sos_adj_record, or the season-to-date
# scoring average that normalises point differential) are not truncated.
#
# 6dp because norm_pts_diff values sit around 0.05 — the 4dp used elsewhere
# would leave them with only three significant figures.
CSV_FLOAT_FORMAT = "%.6f"

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
