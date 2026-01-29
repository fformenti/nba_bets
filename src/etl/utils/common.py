"""Common utility functions for data processing."""

import pandas as pd
from src.config.constants import NEUTRAL_COURT_GAME_LABELS


def get_season_date_range(games):
    all_season_date_range = []
    seasons = games["season"].unique()
    for season in seasons:
        games_season = games.loc[games["season"] == season].copy()
        season_start = games_season["gameDate"].min()
        season_end = games_season["gameDate"].max()

        date_range = pd.date_range(start=season_start, end=season_end)
        season_dates = pd.MultiIndex.from_product(
            [date_range], names=["gameDate"]
        ).to_frame(index=False)
        season_dates["season"] = season
        season_dates["gameDateOnlyStr"] = season_dates["gameDate"].dt.strftime(
            "%Y-%m-%d"
        )
        season_dates.drop(columns=["gameDate"], inplace=True)
        all_season_date_range.append(season_dates)

    return pd.concat(all_season_date_range)


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


def add_neutral_court_game_flag(
    df: pd.DataFrame,
    game_label_column: str = "gameLabel",
    drop_label_column: bool = True,
) -> pd.DataFrame:
    """
    Add a boolean flag indicating if a game is played on a neutral court.

    Neutral court games include international games and Las Vegas games
    as defined in NEUTRAL_COURT_GAME_LABELS.

    Parameters
    ----------
    df : pd.DataFrame
        Games DataFrame with a game label column
    game_label_column : str, default="gameLabel"
        Name of the column containing game labels
    drop_label_column : bool, default=True
        Whether to drop the game label column after processing

    Returns
    -------
    pd.DataFrame
        DataFrame with 'is_neutral_court_game' boolean column added.
        If game_label_column doesn't exist, all values will be False.
    """
    df = df.copy()

    df["is_neutral_court_game"] = df[game_label_column].isin(NEUTRAL_COURT_GAME_LABELS)

    if drop_label_column:
        df = df.drop(columns=[game_label_column]).copy()

    return df
