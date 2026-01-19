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
    lags: List[int],
    location_lags,
) -> pd.DataFrame:
    """
    Create delta features (home - away) for specified feature pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with home and away features
    lags : list
        List of lags to create delta features for
    feature_names : list, optional
        List of feature base names (without prefix). If None, uses common features.

    Returns
    -------
    pd.DataFrame
        DataFrame with delta features added
    """
    record_features = ["record" + "_L" + str(lag) for lag in lags]
    pts_diff_features = ["pts_diff_avg" + "_L" + str(lag) for lag in lags]
    rested_days_features = ["rested_days"]
    feature_names = record_features + pts_diff_features + rested_days_features

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
    feature_names = location_record_features + location_pts_diff_features
    for feature in feature_names:
        home_col = f"{feature}_{HOME_AT_HOME_SUFFIX}"
        away_col = f"{feature}_{AWAY_ON_ROAD_SUFFIX}"

        delta_col = f"{feature}_delta_at_location"
        df[delta_col] = df[home_col] - df[away_col]
        created_features.append(delta_col)
        features_used_for_deltas.extend([home_col, away_col])

    logger.info(f"Created {len(created_features)} delta features")
    df.drop(columns=features_used_for_deltas, inplace=True)
    logger.info(f"Dropped {len(features_used_for_deltas)} features used for deltas")
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
            "east_record_adjusted",
            "west_record_adjusted",
            "hometeamConference_binary",
            "awayteamConference_binary",
            "east_record_adjusted_advantage",
            "conference_diff",
        ]
    )

    logger.info("Created conference_diff_east_pct feature")
    return df


def get_home_conference_vs_away_conference_record(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select east or west record features.
    """
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
            "east_record_at_east",
            "west_record_at_west",
            "games_played_at_east",
            "games_played_at_west",
            "games_played_east_vs_west",
        ]
    )
    return df


def identify_feature_types(
    df: pd.DataFrame,
    exclude_columns: Optional[List[str]] = None,
) -> dict[str, List[str]]:
    """
    Identify numerical and categorical features in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    exclude_columns : list, optional
        List of column names to exclude from feature identification

    Returns
    -------
    dict
        Dictionary with keys 'numerical' and 'categorical' containing lists of column names
    """
    if exclude_columns is None:
        exclude_columns = []

    available_cols = [col for col in df.columns if col not in exclude_columns]

    numerical_features = (
        df[available_cols].select_dtypes(include=["number"]).columns.tolist()
    )

    categorical_features = (
        df[available_cols]
        .select_dtypes(include=["object", "category"])
        .columns.tolist()
    )

    return {
        "numerical": numerical_features,
        "categorical": categorical_features,
    }
