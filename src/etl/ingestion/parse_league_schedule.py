import pandas as pd

from src.config.paths import LEAGUE_SCHEDULE_PATH, PROCESSED_LEAGUE_SCHEDULE_PATH
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)
setup_logging(level="INFO")


def parse_schedule(df):
    # date_time_est = "2025-10-02 12:00:00Z"
    # date = pd.to_datetime(date_time_est).tz_convert(None)
    # df["gameDate"] = pd.to_datetime(df["gameDateTimeEst"].tz_convert(None))
    df = df[~df["gameLabel"].isin(["Preseason"])]
    df = df[(df["homeTeamId"] != 0) & (df["awayTeamId"] != 0)]
    return df


def build_processed_schedule() -> None:
    """Parse the league schedule CSV into its processed form."""
    raw_schedule = pd.read_csv(
        LEAGUE_SCHEDULE_PATH, parse_dates=["gameDateTimeEst"], low_memory=False
    )
    parsed_schedule = parse_schedule(raw_schedule)
    parsed_schedule.to_csv(PROCESSED_LEAGUE_SCHEDULE_PATH, index=False)
    logger.info(
        f"Saved {len(parsed_schedule)} parsed games to {PROCESSED_LEAGUE_SCHEDULE_PATH}"
    )
