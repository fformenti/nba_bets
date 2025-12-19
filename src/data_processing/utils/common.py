"""Common utility functions for data processing."""

import pandas as pd


def calculate_arena_occupation(home_games):
    """
    Calculate arena occupation metrics for home games.

    Parameters
    ----------
    home_games : pd.DataFrame
        DataFrame with home games data including attendance

    Returns
    -------
    pd.DataFrame
        DataFrame with arena occupation metrics added
    """
    home_games["attendance_last_game"] = home_games.groupby(["teamId"])[
        "attendance"
    ].shift(1)
    home_games["attendance_last_5"] = (
        home_games.groupby("teamId")["attendance_last_game"]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    home_games["attendance_last_game"] = home_games.groupby("teamId")[
        "attendance_last_game"
    ].bfill()
    return home_games


def get_nba_season(game_date):
    """
    Determine the NBA season for a given date.

    NBA season spans October-June, labeled as YYYY/YY where October is in the first year.
    Example: Feb 25, 2025 → season "2024/25" (because season started Oct 2024)

    Parameters
    ----------
    game_date : pd.Timestamp or datetime
        Game date

    Returns
    -------
    str
        Season string in format "YYYY/YY"
    """
    year = game_date.year
    month = game_date.month

    # If month is October or later, season is current_year/next_year
    if month >= 9:
        return f"{year}/{(year + 1) % 100:02d}"
    # Otherwise season is previous_year/current_year
    else:
        return f"{year - 1}/{year % 100:02d}"


def filter_games_by_date(games: pd.DataFrame, min_date: str) -> pd.DataFrame:
    """
    Filter games DataFrame to include only games on or after the specified date.

    Parameters
    ----------
    games : pd.DataFrame
        Games DataFrame with a 'gameDate' column
    min_date : str
        Minimum date to filter games (format: "YYYY-MM-DD").
        Only games on or after this date will be included.

    Returns
    -------
    pd.DataFrame
        Filtered games DataFrame

    Raises
    ------
    ValueError
        If 'gameDate' column is not present in the DataFrame
    """
    if "gameDate" not in games.columns:
        raise ValueError("DataFrame must contain 'gameDate' column")

    min_date_dt = pd.to_datetime(min_date)
    initial_count = len(games)
    filtered_games = games[games["gameDate"] >= min_date_dt].copy()
    filtered_count = len(filtered_games)

    print(f"Filtered games: {initial_count} -> {filtered_count} (min_date: {min_date})")

    return filtered_games
