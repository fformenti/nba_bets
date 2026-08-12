"""
Transform teams history table to expand each team's history into individual seasons.

This script reads the TeamsHistoriesConferenceNBA.csv file and creates a new table
where each row represents a team in a specific season, expanding the seasonFounded
to seasonActiveTill range into individual season rows.
"""

from pathlib import Path

import pandas as pd

from src.config.constants import CURRENT_SEASON_START_YEAR
from src.config.paths import (
    PROCESSED_DIR,
    TEAMS_CITIES_CONFERENCE_HISTORY_HANDMADE_PATH,
    TEAMS_CITIES_CONFERENCE_HISTORY_PROCESSED_PATH,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def format_season(year: int) -> str:
    """
    Format a year into season format (e.g., 2002 -> "2002/03").

    Args:
        year: The starting year of the season

    Returns:
        Formatted season string (YYYY/YY)
    """
    next_year = year + 1
    return f"{year}/{str(next_year)[-2:]}"


def create_teams_history_table(
    input_file: Path | str,
    output_file: Path | str | None = None,
    current_season_start_year: int = CURRENT_SEASON_START_YEAR,
) -> pd.DataFrame:
    """
    Transform teams history table to expand seasons.

    Args:
        input_file: Path to input CSV file
        output_file: Optional path to save output CSV file
        current_season_year: Current season year (default: 2024 for 2024/25 season)

    Returns:
        DataFrame with columns: teamId, teamCity, teamName, teamAbbrev, season, Conference
    """
    # Read the input CSV
    df = pd.read_csv(input_file)

    # Filter: league == "NBA", teamCity != "All-star"
    df = df[df["league"] == "NBA"]
    df = df[df["teamCity"] != "All-star"]

    # Overwrite seasonActiveTill == 2100 to current_season_year + 1
    # (2025 for 2024/25 season as per user instruction)
    max_season_year = current_season_start_year + 1
    df.loc[df["seasonActiveTill"] == 2100, "seasonActiveTill"] = max_season_year

    # Create list to store expanded rows
    expanded_rows = []

    # Group by teamId to handle gaps properly
    for team_id in df["teamId"].unique():
        team_df = df[df["teamId"] == team_id].sort_values("seasonFounded")

        for _, row in team_df.iterrows():
            season_founded = int(row["seasonFounded"])
            season_active_till = int(row["seasonActiveTill"])

            # Generate seasons from seasonFounded to seasonActiveTill (exclusive)
            # seasonActiveTill represents the first year of the next period
            # But cap at current_season_year + 1 to ensure latest season is 2024/25
            end_year = min(season_active_till, current_season_start_year + 1)
            for year in range(season_founded, end_year):
                season_str = format_season(year)

                expanded_rows.append(
                    {
                        "teamId": row["teamId"],
                        "teamCity": row["teamCity"],
                        "teamName": row["teamName"],
                        "teamAbbrev": row["teamAbbrev"].strip(),  # Remove trailing spaces
                        "season": season_str,
                        "Conference": row["Conference"],
                    }
                )

    # Create DataFrame from expanded rows
    result_df = pd.DataFrame(expanded_rows)

    # Remove duplicates (in case there are overlapping periods)
    result_df = result_df.drop_duplicates(subset=["teamId", "season"], keep="first")

    # Sort by teamId and season
    result_df = result_df.sort_values(["teamId", "season"]).reset_index(drop=True)

    # Save to file if output_file is provided
    if output_file:
        result_df.to_csv(output_file, index=False)
        print(f"Output saved to {output_file}")

    return result_df


def load_teams_history_table(
    processed_file: Path | str | None = None,
) -> pd.DataFrame:
    """
    Load the precomputed teams history table from disk.

    Raises a FileNotFoundError if the file does not exist.
    """
    if processed_file is None:
        processed_file = str(TEAMS_CITIES_CONFERENCE_HISTORY_PROCESSED_PATH)
    if not Path(processed_file).exists():
        raise FileNotFoundError(
            "Teams history table not found. Run src/etl/ingestion/teams_history.py "
            "to generate it first."
        )
    return pd.read_csv(processed_file)


def build_teams_history() -> None:
    """Build the team/city/conference history lookup that features join against."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    result = create_teams_history_table(
        input_file=TEAMS_CITIES_CONFERENCE_HISTORY_HANDMADE_PATH,
        output_file=TEAMS_CITIES_CONFERENCE_HISTORY_PROCESSED_PATH,
        current_season_start_year=CURRENT_SEASON_START_YEAR,
    )
    logger.info(f"Processed {len(result)} rows")
    logger.info(f"First few rows:\n{result.head(10)}")
    logger.info(f"Last few rows:\n{result.tail(10)}")
