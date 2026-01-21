"""
Transform teams history table to expand each team's history into individual seasons.

This script reads the TeamsHistoriesConferenceNBA.csv file and creates a new table
where each row represents a team in a specific season, expanding the seasonFounded
to seasonActiveTill range into individual season rows.
"""

import pandas as pd
from pathlib import Path


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
    input_file: str, output_file: str = None, current_season_year: int = 2024
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
    max_season_year = current_season_year + 1
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
            end_year = min(season_active_till, current_season_year + 1)
            for year in range(season_founded, end_year):
                season_str = format_season(year)

                expanded_rows.append(
                    {
                        "teamId": row["teamId"],
                        "teamCity": row["teamCity"],
                        "teamName": row["teamName"],
                        "teamAbbrev": row[
                            "teamAbbrev"
                        ].strip(),  # Remove trailing spaces
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


if __name__ == "__main__":
    # Define paths
    project_root = Path(__file__).parent.parent.parent.parent
    input_file = (
        project_root / "data" / "raw" / "historical" / "TeamsHistoriesConferenceNBA.csv"
    )

    # Create output directory if it doesn't exist
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "teams_history_expanded.csv"

    # Process the data
    result = create_teams_history_table(
        input_file=str(input_file),
        output_file=str(output_file),
        current_season_year=2024,
    )

    print(f"\nProcessed {len(result)} rows")
    print("\nFirst few rows:")
    print(result.head(10))
    print("\nLast few rows:")
    print(result.tail(10))
