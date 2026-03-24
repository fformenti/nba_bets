"""Common utility functions for data processing."""

import pandas as pd
from src.config.constants import NEUTRAL_COURT_GAME_LABELS
from src.config.paths import TEAMS_CITIES_LOCATIONS_HISTORY_PROCESSED_PATH


CANONICAL_INGESTED_COLUMNS = [
    "gameId",
    "gameDate",
    "gameDateOnlyStr",
    "season",
    "hometeamPrename",
    "hometeamName",
    "hometeamId",
    "awayteamPrename",
    "awayteamName",
    "awayteamId",
    "homeScore",
    "awayScore",
    "winner",
    "overtimes",
    "postponed",
    "gameType",
    "attendance",
    "arenaId",
    "gameLabel",
    "gameSubLabel",
    "seriesGameNumber",
    "hometeamLocation",
    "gameLocation",
    "awayteamLocation",
]


def get_teams_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich a games DataFrame with home/away/game location columns via teams history lookup."""
    teams_cities_states = pd.read_csv(TEAMS_CITIES_LOCATIONS_HISTORY_PROCESSED_PATH)
    # Home team location
    df = df.merge(
        teams_cities_states[["teamId", "season", "city", "state"]],
        left_on=["hometeamId", "season"],
        right_on=["teamId", "season"],
        how="left",
    ).drop(columns=["teamId"])
    df["hometeamLocation"] = df["city"] + ", " + df["state"]
    df.drop(columns=["city", "state"], inplace=True)

    # Game Location (To do: in the future it should reference arenaId)
    df["gameLocation"] = df["hometeamLocation"]

    # Away team location
    df = df.merge(
        teams_cities_states[["teamId", "season", "city", "state"]],
        left_on=["awayteamId", "season"],
        right_on=["teamId", "season"],
        how="left",
    ).drop(columns=["teamId"])
    df["awayteamLocation"] = df["city"] + ", " + df["state"]
    df.drop(columns=["city", "state"], inplace=True)
    return df


def enrich_games_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize city names and add location columns if missing or incomplete."""
    normalize_city_names = {"LA": "Los Angeles"}
    if "hometeamPrename" in df.columns:
        df["hometeamPrename"] = df["hometeamPrename"].replace(normalize_city_names)
    if "awayteamPrename" in df.columns:
        df["awayteamPrename"] = df["awayteamPrename"].replace(normalize_city_names)

    location_cols = ["hometeamLocation", "awayteamLocation", "gameLocation"]
    needs_enrichment = any(col not in df.columns for col in location_cols)
    if not needs_enrichment:
        needs_enrichment = df[location_cols].isna().any().any()

    if needs_enrichment:
        # Drop any partial location columns before re-merging
        existing_loc_cols = [c for c in location_cols if c in df.columns]
        if existing_loc_cols:
            df = df.drop(columns=existing_loc_cols)
        df = get_teams_locations(df)

    return df


def deduplicate_games(
    existing_df: pd.DataFrame, new_df: pd.DataFrame
) -> pd.DataFrame:
    """Remove rows from existing_df that match any row in new_df on composite key."""
    key_cols = ["gameDateOnlyStr", "hometeamId", "awayteamId"]
    new_keys = set(
        tuple(row)
        for row in new_df[key_cols].dropna().itertuples(index=False)
    )
    existing_keys = existing_df[key_cols].dropna().itertuples(index=False)
    mask = [tuple(row) not in new_keys for row in existing_keys]
    return existing_df.loc[mask].copy()


def coerce_numeric_columns(
    df: pd.DataFrame, columns: list[str], dtype: str = "int64"
) -> pd.DataFrame:
    """Convert columns to numeric types, coercing errors to NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (modified in place).
    columns : list[str]
        Column names to convert.
    dtype : str, default="int64"
        Target dtype after conversion. Use "Int64" for nullable integers.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
    return df


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
