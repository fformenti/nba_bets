import pandas as pd

from src.config.paths import LEAGUE_SCHEDULE_PATH, PROCESSED_LEAGUE_SCHEDULE_PATH
from src.etl.utils.common import atomic_write_csv, game_type_from_id
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)
setup_logging(level="INFO")

# Fixtures worth collecting. All-Star weekend is excluded: the Rising Stars and
# the exhibition itself are not games between two NBA teams, but they carry
# gameIds and team placeholders, so nothing else keeps them out — they were
# reaching the history table as postponed rows with no teams.
COLLECTABLE_GAME_TYPES = {"Regular Season", "Playoffs", "Play-in Tournament"}


def parse_schedule(df):
    # date_time_est = "2025-10-02 12:00:00Z"
    # date = pd.to_datetime(date_time_est).tz_convert(None)
    # df["gameDate"] = pd.to_datetime(df["gameDateTimeEst"].tz_convert(None))
    game_types = df["gameId"].map(game_type_from_id)
    df = df[game_types.isin(COLLECTABLE_GAME_TYPES)]
    df = df[(df["homeTeamId"] != 0) & (df["awayTeamId"] != 0)]
    return df


def build_processed_schedule() -> None:
    """Parse the league schedule CSV into its processed form."""
    raw_schedule = pd.read_csv(
        LEAGUE_SCHEDULE_PATH, parse_dates=["gameDateTimeEst"], low_memory=False
    )
    parsed_schedule = parse_schedule(raw_schedule)
    # Atomic: the collector prunes its pending queue against this file, so a
    # truncated copy would silently delete payloads for games it no longer sees.
    atomic_write_csv(parsed_schedule, PROCESSED_LEAGUE_SCHEDULE_PATH)
    logger.info(
        f"Saved {len(parsed_schedule)} parsed games to {PROCESSED_LEAGUE_SCHEDULE_PATH}"
    )
