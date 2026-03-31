"""Feature engineering utilities for NBA games data."""

import logging
from typing import List, Optional

import pandas as pd

from src.ml.config.schema import FeaturesMapConfig, MomentumPairConfig

logger = logging.getLogger(__name__)


HOME_SUFFIX = "HT"
AWAY_SUFFIX = "VT"
HOME_AT_HOME_SUFFIX = f"{HOME_SUFFIX}_at_home"
AWAY_ON_ROAD_SUFFIX = f"{AWAY_SUFFIX}_on_road"

# Column prefix mapping for each feature group.
# Each entry: (group_name, prefix_fn, has_location_variant)
# prefix_fn takes lags and returns list of column base names.
FEATURE_GROUP_PREFIXES = {
    "record": lambda lags: [f"record_L{lag}" for lag in lags],
    "point_differential": lambda lags: [f"pts_diff_avg_L{lag}" for lag in lags],
    "sos": lambda lags: [f"sos_L{lag}" for lag in lags],
    "sos_adj_record": lambda lags: [f"sos_adj_record_L{lag}" for lag in lags],
    "distance": lambda lags: [f"distance_L{lag}" for lag in lags],
    "rested_days": lambda _: ["rested_days"],
    "streak": lambda _: ["streak"],
    "last_season_record": lambda _: ["last_season_record"],
}

# Groups that have HT_at_home / VT_on_road location variants
LOCATION_VARIANT_GROUPS = {"record", "point_differential", "last_season_record", "sos_adj_record"}


def create_delta_features(
    df: pd.DataFrame,
    features_config: FeaturesMapConfig,
) -> pd.DataFrame:
    """
    Create delta features (home - away) based on feature group configuration.

    For each enabled feature group with ``delta=True``, creates
    ``{prefix}_delta = {prefix}_HT - {prefix}_VT``.  Source HT/VT columns
    are always dropped after delta creation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with home and away features.
    features_config : FeaturesMapConfig
        Feature group configuration controlling lags, deltas, and drops.

    Returns
    -------
    pd.DataFrame
        DataFrame with delta features added and originals dropped.
    """
    df = df.copy()
    created_features: list[str] = []
    columns_to_drop: list[str] = []

    for group_name, prefix_fn in FEATURE_GROUP_PREFIXES.items():
        group_config = getattr(features_config, group_name)
        prefixes = prefix_fn(group_config.lags)

        if not group_config.enabled:
            for prefix in prefixes:
                columns_to_drop.extend([f"{prefix}_{HOME_SUFFIX}", f"{prefix}_{AWAY_SUFFIX}"])
            if group_name in LOCATION_VARIANT_GROUPS:
                loc_prefixes = prefix_fn(group_config.location_lags)
                for prefix in loc_prefixes:
                    columns_to_drop.extend(
                        [f"{prefix}_{HOME_AT_HOME_SUFFIX}", f"{prefix}_{AWAY_ON_ROAD_SUFFIX}"]
                    )
            continue

        if group_config.delta:
            for prefix in prefixes:
                home_col = f"{prefix}_{HOME_SUFFIX}"
                away_col = f"{prefix}_{AWAY_SUFFIX}"
                if home_col not in df.columns or away_col not in df.columns:
                    logger.debug(f"Skipping delta for {prefix}: columns not found")
                    continue
                df[f"{prefix}_delta"] = df[home_col] - df[away_col]
                created_features.append(f"{prefix}_delta")
                columns_to_drop.extend([home_col, away_col])

            if group_name in LOCATION_VARIANT_GROUPS:
                loc_prefixes = prefix_fn(group_config.location_lags)
                for prefix in loc_prefixes:
                    home_col = f"{prefix}_{HOME_AT_HOME_SUFFIX}"
                    away_col = f"{prefix}_{AWAY_ON_ROAD_SUFFIX}"
                    if home_col not in df.columns or away_col not in df.columns:
                        logger.debug(f"Skipping location delta for {prefix}: columns not found")
                        continue
                    df[f"{prefix}_at_location_delta"] = df[home_col] - df[away_col]
                    created_features.append(f"{prefix}_at_location_delta")
                    columns_to_drop.extend([home_col, away_col])
        # delta=False + enabled=True → raw HT/VT survive, nothing to drop

    # home_and_road: special case — delta is a SUM, not a difference
    home_road_config = features_config.home_and_road
    if not home_road_config.enabled:
        columns_to_drop.extend(["days_at_home", "days_on_road"])
    elif home_road_config.delta:
        df["days_at_home_delta"] = df["days_at_home"] + df["days_on_road"]
        created_features.append("days_at_home_delta")
        columns_to_drop.extend(["days_at_home", "days_on_road"])
    # else: delta=False → raw survive

    # Drop columns
    cols_to_drop = [c for c in columns_to_drop if c in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        logger.info(f"Dropped {len(cols_to_drop)} columns after delta creation")

    logger.info(f"Created {len(created_features)} delta features")
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


def resolve_feature_columns(
    features_config: FeaturesMapConfig,
    conference_filter: str,
    momentum_pairs: Optional[List[MomentumPairConfig]] = None,
) -> list[str]:
    """
    Compute the list of feature columns expected after delta creation and conference
    feature engineering, based on the feature group configuration.

    Used in inclusion mode to select exactly the declared features from the DataFrame.
    Returns column names in a deterministic order. Columns that are expected but not
    present in the DataFrame will be flagged by the caller.

    Parameters
    ----------
    features_config : FeaturesMapConfig
        Feature group configuration.
    conference_filter : str
        Conference filter type: 'same', 'different', or 'all'.
    momentum_pairs : list of MomentumPairConfig, optional
        Momentum pairs that replace two delta columns with one momentum column.

    Returns
    -------
    list[str]
        Ordered list of expected feature column names.
    """
    columns: list[str] = []

    for group_name, prefix_fn in FEATURE_GROUP_PREFIXES.items():
        group_config = getattr(features_config, group_name)

        if not group_config.enabled:
            continue

        prefixes = prefix_fn(group_config.lags)

        if group_config.delta:
            for prefix in prefixes:
                columns.append(f"{prefix}_delta")
            if group_name in LOCATION_VARIANT_GROUPS:
                loc_prefixes = prefix_fn(group_config.location_lags)
                for prefix in loc_prefixes:
                    columns.append(f"{prefix}_at_location_delta")
        else:
            # delta=False + enabled=True → raw HT/VT survive
            for prefix in prefixes:
                columns.extend([f"{prefix}_{HOME_SUFFIX}", f"{prefix}_{AWAY_SUFFIX}"])

    # home_and_road special case
    home_road_config = features_config.home_and_road
    if home_road_config.enabled:
        if home_road_config.delta:
            columns.append("days_at_home_delta")
        else:
            columns.extend(["days_at_home", "days_on_road"])

    # Conference features (created dynamically by apply_conference_features)
    if conference_filter == "different":
        columns.extend(
            [
                "home_conference_vs_away_conference_record",
                "games_played_at_home_conference",
            ]
        )
    elif conference_filter == "all":
        columns.append("conference_diff_home_advantage_pct")

    # Momentum pairs: replace two source deltas with one momentum column
    if momentum_pairs:
        for pair in momentum_pairs:
            short_col = f"{pair.feature}_L{pair.short}_delta"
            long_col = f"{pair.feature}_L{pair.long}_delta"
            momentum_col = f"{pair.feature}_momentum_L{pair.short}_L{pair.long}_delta"
            if short_col in columns:
                columns.remove(short_col)
            if long_col in columns:
                columns.remove(long_col)
            columns.append(momentum_col)

    return columns


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
