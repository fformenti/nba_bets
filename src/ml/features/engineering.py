"""Feature engineering utilities for NBA games data."""

import logging
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


HOME_SUFFIX = "HT"
AWAY_SUFFIX = "VT"
HOME_AT_HOME_SUFFIX = f"{HOME_SUFFIX}_at_home"
AWAY_ON_ROAD_SUFFIX = f"{AWAY_SUFFIX}_on_road"


def create_delta_features(
    df: pd.DataFrame,
    record_lags: List[int],
    point_differential_lags: List[int],
    location_lags,
    distances_lags,
) -> pd.DataFrame:
    """
    Create delta features (home - away) for specified feature pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with home and away features
    record_lags : list
        List of lags to create record delta features for
    point_differential_lags : list
        List of lags to create point differential delta features for

    Returns
    -------
    pd.DataFrame
        DataFrame with delta features added
    """
    record_features = ["record" + "_L" + str(lag) for lag in record_lags]
    pts_diff_features = [
        "pts_diff_avg" + "_L" + str(lag) for lag in point_differential_lags
    ]
    distances_features = ["distance_L" + str(lag) for lag in distances_lags]
    rested_days_features = ["rested_days"]
    last_season_record_features = ["last_season_record"]
    streak_features = ["streak"]
    feature_names = (
        record_features
        + pts_diff_features
        + distances_features
        + rested_days_features
        + last_season_record_features
        + streak_features
    )

    df = df.copy()
    created_features = []
    features_used_for_deltas = []
    # General features lag features
    for feature in feature_names:
        home_col = f"{feature}_{HOME_SUFFIX}"
        away_col = f"{feature}_{AWAY_SUFFIX}"

        delta_col = f"{feature}_delta"
        df[delta_col] = df[home_col] - df[away_col]
        created_features.append(delta_col)
        features_used_for_deltas.extend([home_col, away_col])

    # Location specific features lag features
    location_record_features = ["record" + "_L" + str(lag) for lag in location_lags]
    location_pts_diff_features = [
        "pts_diff_avg" + "_L" + str(lag) for lag in location_lags
    ]
    feature_names = (
        location_record_features
        + location_pts_diff_features
        + last_season_record_features
    )
    for feature in feature_names:
        home_col = f"{feature}_{HOME_AT_HOME_SUFFIX}"
        away_col = f"{feature}_{AWAY_ON_ROAD_SUFFIX}"

        delta_col = f"{feature}_delta_at_location"
        df[delta_col] = df[home_col] - df[away_col]
        created_features.append(delta_col)
        features_used_for_deltas.extend([home_col, away_col])

    # Delta between days at home and days on road
    df["days_at_home_delta"] = df["days_at_home"] + df["days_on_road"]
    created_features.append("days_at_home_delta")
    features_used_for_deltas.extend(["days_at_home", "days_on_road"])

    logger.info(f"Created {len(created_features)} delta features")
    df.drop(columns=features_used_for_deltas, inplace=True)
    logger.info(f"Dropped {len(features_used_for_deltas)} features used for deltas")
    return df


def create_momentum_features(
    df: pd.DataFrame,
    momentum_pairs: List[List[int]],
) -> pd.DataFrame:
    """
    Replace correlated lag-delta pairs with a single momentum feature.

    For each [short_lag, long_lag] pair, computes:
        pts_diff_momentum_L{short}_L{long}_delta = pts_diff_avg_L{short}_delta - pts_diff_avg_L{long}_delta

    Both source delta columns are dropped.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with delta features already created
    momentum_pairs : list of [short_lag, long_lag]
        Each pair defines a momentum feature replacing the two correlated lag deltas

    Returns
    -------
    pd.DataFrame
        DataFrame with momentum features added and source lag deltas dropped
    """
    df = df.copy()
    for pair in momentum_pairs:
        feature, short_lag, long_lag = pair.feature, pair.short, pair.long
        short_col = f"{feature}_L{short_lag}_delta"
        long_col = f"{feature}_L{long_lag}_delta"
        momentum_col = f"{feature}_momentum_L{short_lag}_L{long_lag}_delta"
        df[momentum_col] = df[short_col] - df[long_col]
        df.drop(columns=[short_col, long_col], inplace=True)
        logger.info(f"Created {momentum_col}, dropped {short_col} and {long_col}")
    return df


def create_conference_delta(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create conference-based feature: difference between binary conference values * east_wins_pct.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with conference columns
    home_conf_col : str, default='hometeamConference'
        Home team conference column name
    away_conf_col : str, default='awayteamConference'
        Away team conference column name
    east_wins_pct_col : str, default='east_wins_pct_L1'
        East wins percentage column name

    Returns
    -------
    pd.DataFrame
        DataFrame with conference_diff_east_pct feature added
    """
    df = df.copy()

    # Encode conferences as binary: East=1, West=0
    home_conf_col: str = "hometeamConference"
    away_conf_col: str = "awayteamConference"
    df["hometeamConference_binary"] = df[home_conf_col].map({"East": 1, "West": 0})
    df["awayteamConference_binary"] = df[away_conf_col].map({"East": 1, "West": 0})

    df["east_record_adjusted_advantage"] = (
        df["east_record_adjusted"] - df["west_record_adjusted"]
    )

    # East vs West = 1, West vs East = -1
    df["conference_diff"] = (
        df["hometeamConference_binary"] - df["awayteamConference_binary"]
    )

    # Positive values means the home team has advantage over the away team, negative means the away team has advantage over the home team.
    df["conference_diff_home_advantage_pct"] = (
        df["conference_diff"] * df["east_record_adjusted_advantage"]
    )

    # Drop intermediate columns
    df = df.drop(
        columns=[
            "hometeamConference",
            "awayteamConference",
            "east_record_adjusted",
            "west_record_adjusted",
            "hometeamConference_binary",
            "awayteamConference_binary",
            "east_record_adjusted_advantage",
            "conference_diff",
        ]
    )

    logger.info(
        "Created conference_diff_home_advantage_pct feature (0.0 for same conference teams)"
    )
    return df


def get_home_conference_vs_away_conference_record(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create conference-specific features for different conference matchups.

    Creates:
    - home_conference_vs_away_conference_record: Record of home team's conference
      when playing at home conference
    - games_played_at_home_conference: Games played by home team at their conference

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with conference columns and east/west record features

    Returns
    -------
    pd.DataFrame
        DataFrame with conference features added and intermediate columns dropped
    """
    df = df.copy()

    df["home_conference_vs_away_conference_record"] = df.apply(
        lambda x: x["east_record_at_east"]
        if x["hometeamConference"] == "East"
        else x["west_record_at_west"],
        axis=1,
    )

    df["games_played_at_home_conference"] = df.apply(
        lambda x: x["games_played_at_east"]
        if x["hometeamConference"] == "East"
        else x["games_played_at_west"],
        axis=1,
    )

    df = df.drop(
        columns=[
            "east_record_adjusted",
            "west_record_adjusted",
            "east_record_at_east",
            "west_record_at_west",
            "games_played_at_east",
            "games_played_at_west",
            "games_played_east_vs_west",
        ]
    )

    logger.info(
        "Created conference features: home_conference_vs_away_conference_record, "
        "games_played_at_home_conference"
    )
    return df


def apply_conference_features(
    df: pd.DataFrame,
    conference_filter: str,
) -> pd.DataFrame:
    """
    Apply appropriate conference features based on conference filter type.

    This function ensures consistent feature engineering across training and prediction:
    - 'same': No conference features (teams from same conference)
    - 'different': Conference vs conference record features (teams from different conferences)
    - 'all': Conference delta feature (works for both same and different conferences,
             equals 0.0 for same conference matchups)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with delta features already created
    conference_filter : str
        Conference filter type: 'same', 'different', or 'all'

    Returns
    -------
    pd.DataFrame
        DataFrame with appropriate conference features added

    Raises
    ------
    ValueError
        If conference_filter is not one of the valid options
    """
    if conference_filter not in ["same", "different", "all"]:
        raise ValueError(
            f"conference_filter must be 'same', 'different', or 'all', got '{conference_filter}'"
        )

    df = df.copy()

    if conference_filter == "same":
        # Same conference teams: no conference features needed
        logger.info("Same conference filter: skipping conference features")
        # Drop any conference-related columns that might exist
        conference_cols_to_drop = [
            "east_record_adjusted",
            "west_record_adjusted",
            "east_record_at_east",
            "west_record_at_west",
            "games_played_at_east",
            "games_played_at_west",
            "games_played_east_vs_west",
            "home_conference_vs_away_conference_record",
            "games_played_at_home_conference",
            "conference_diff_home_advantage_pct",
        ]
        df = df.drop(
            columns=[col for col in conference_cols_to_drop if col in df.columns]
        )

    elif conference_filter == "different":
        # Different conference teams: use conference vs conference record features
        logger.info(
            "Different conference filter: adding conference vs conference record features"
        )
        df = get_home_conference_vs_away_conference_record(df)

    elif conference_filter == "all":
        # All teams: use conference delta feature (0.0 for same conference)
        logger.info("All teams filter: adding conference delta feature")
        df = create_conference_delta(df)

    return df


def identify_feature_types(
    df: pd.DataFrame,
    exclude_columns: Optional[List[str]] = None,
) -> dict[str, List[str]]:
    """
    Identify numerical, categorical, and boolean features in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    exclude_columns : list, optional
        List of column names to exclude from feature identification

    Returns
    -------
    dict
        Dictionary with keys 'numerical', 'categorical', and 'boolean'
        containing lists of column names
    """
    if exclude_columns is None:
        exclude_columns = []

    available_cols = [col for col in df.columns if col not in exclude_columns]

    # Identify boolean columns (binary categorical features)
    boolean_features = (
        df[available_cols].select_dtypes(include=["bool"]).columns.tolist()
    )

    # Numerical features: get all numeric columns, then exclude boolean columns
    all_numeric = df[available_cols].select_dtypes(include=["number"]).columns.tolist()
    numerical_features = [col for col in all_numeric if col not in boolean_features]

    # Categorical features: include object and category (but NOT boolean)
    categorical_features = (
        df[available_cols]
        .select_dtypes(include=["object", "category"])
        .columns.tolist()
    )

    return {
        "numerical": numerical_features,
        "categorical": categorical_features,
        "boolean": boolean_features,
    }
