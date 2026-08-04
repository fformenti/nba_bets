"""Feature construction for upcoming-game prediction.

Mirrors the training-time feature path: the same ``create_features_tables`` /
``merge_features`` from the ETL layer, then the same delta and conference
feature engineering. Upcoming games are concatenated onto history first, because
rolling features (records, point differentials, rest) are only defined relative
to the games that came before.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.config.paths import DEFAULT_FEATURES_CONFIG_PATH
from src.etl.features.aggregator import (
    create_features_tables_from_config,
    merge_features,
)
from src.etl.utils.common import get_nba_season
from src.ml.config.loader import load_features_config
from src.ml.config.schema import ExperimentConfig
from src.ml.features.engineering import (
    apply_conference_features,
    create_delta_features,
    resolve_feature_columns,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def fix_upcoming_games_cols(
    upcoming_games: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare upcoming games DataFrame with required columns.

    Parameters
    ----------
    upcoming : pd.DataFrame
        Raw upcoming games DataFrame
    teams_history : pd.DataFrame
        Teams history DataFrame with conferences

    Returns
    -------
    pd.DataFrame
        Prepared upcoming games DataFrame
    """
    logger.info(f"Preparing {len(upcoming_games)} upcoming games")

    df = upcoming_games.copy()
    if "gameDateOnlyStr" not in df.columns and "gameDate" in df.columns:
        df["gameDateOnlyStr"] = pd.to_datetime(df["gameDate"], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )
    if "season" not in df.columns and "gameDate" in df.columns:
        df["season"] = pd.to_datetime(df["gameDate"], errors="coerce").apply(get_nba_season)
    columns_to_drop = ["arenaCity", "arenaName"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Set placeholder values for upcoming games
    df["postponed"] = 0
    df["overtimes"] = None
    df["winner"] = 0
    df["homeScore"] = 0
    df["awayScore"] = 0
    df["arenaId"] = None
    df["attendance"] = None
    df["gameType"] = "Regular Season"

    return df


def filter_by_conference(
    df: pd.DataFrame,
    conference_filter: str,
) -> pd.DataFrame:
    """
    Filter games by conference matchup type.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with conference columns
    conference_filter : str
        Filter type: 'different', 'same', or 'all'

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    if conference_filter == "different":
        filtered = df[df["hometeamConference"] != df["awayteamConference"]].copy()
        logger.info(f"Filtered to {len(filtered)} games (different conferences)")
    elif conference_filter == "same":
        filtered = df[df["hometeamConference"] == df["awayteamConference"]].copy()
        logger.info(f"Filtered to {len(filtered)} games (same conference)")
    else:
        filtered = df.copy()
        logger.info(f"No conference filter applied ({len(filtered)} games)")

    return filtered


def build_features_for_prediction(
    upcoming_games: pd.DataFrame,
    historical_features: pd.DataFrame,
    experiment_config: ExperimentConfig,
    conference_filter: str,
    features_config=None,
) -> pd.DataFrame:
    """
    Build features for upcoming games prediction.

    Two configs, two jobs. The **ETL** features config decides which feature
    *tables* get built — it must match the one that produced the historical
    rows. The **experiment** config then decides which of those columns the
    model consumes (delta features, conference features, selection).

    Conflating them was a bug: an experiment may set ``record.lags: []`` while
    leaving a derived group like ``sos_adj_record`` enabled, which is coherent
    for feature selection but incoherent as an ETL instruction, and blew up as
    ``KeyError: 'record_L5'`` inside the SOS-adjusted record builder.

    Parameters
    ----------
    upcoming_games : pd.DataFrame
        Prepared upcoming games DataFrame
    historical_features : pd.DataFrame
        Historical features DataFrame
    experiment_config : ExperimentConfig
        Experiment configuration, for the model-side feature engineering
    conference_filter : str
        Conference filter type ('same', 'different', or 'all').
        Determines which conference features to create:
        - 'same': No conference features
        - 'different': Conference vs conference record features
        - 'all': Conference delta feature (0.0 for same conference)
    features_config : optional
        ETL features config. Defaults to ``configs/features.yaml`` — the same
        file `make build-features` uses.

    Returns
    -------
    pd.DataFrame
        DataFrame with features ready for prediction
    """
    logger.info("Building features for prediction")

    if features_config is None:
        features_config = load_features_config(DEFAULT_FEATURES_CONFIG_PATH)

    # Rolling features are defined relative to prior games, so upcoming rows
    # have to sit on top of history to be computable at all.
    historical_combined = pd.concat([historical_features, upcoming_games])

    feat_eng_config = experiment_config.feature_engineering

    logger.info("Creating feature tables")
    create_features_tables_from_config(historical_combined, features_config)

    # Merge features for upcoming games
    upcoming_with_features = merge_features(upcoming_games)

    # Drop columns that are not needed for feature engineering
    # These columns are added for merge_features but shouldn't be in final features
    columns_to_drop_after_merge = ["winner", "winnerteamConference"]
    upcoming_with_features = upcoming_with_features.drop(
        columns=[
            col for col in columns_to_drop_after_merge if col in upcoming_with_features.columns
        ]
    )

    # Apply conference filter
    upcoming_with_features = filter_by_conference(upcoming_with_features, conference_filter)

    # Create delta features
    logger.info("Creating delta features")
    upcoming_with_features = create_delta_features(
        upcoming_with_features,
        feat_eng_config.features,
    )

    # Apply conference-specific features based on filter type
    # This matches the training logic exactly - conference_filter determines features
    upcoming_with_features = apply_conference_features(upcoming_with_features, conference_filter)

    return upcoming_with_features


def prepare_features_for_model(
    df: pd.DataFrame,
    experiment_config: ExperimentConfig,
    target_column: str,
    conference_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Prepare features DataFrame for model prediction.

    In inclusion mode, only columns declared in the experiment config are kept.
    In exclusion mode (legacy), metadata and intermediate columns are dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with all columns
    experiment_config : ExperimentConfig
        Experiment configuration
    target_column : str
        Target column name
    conference_filter : str, optional
        Conference filter used to resolve conference feature columns. Falls back
        to experiment_config.filters.conference_filter when not provided.

    Returns
    -------
    pd.DataFrame
        Features DataFrame ready for model prediction
    """
    feat_eng_config = experiment_config.feature_engineering

    if feat_eng_config.selection_mode == "inclusion":
        cf = conference_filter or experiment_config.filters.conference_filter
        feature_columns = resolve_feature_columns(feat_eng_config.features, cf)
        available = [c for c in feature_columns if c in df.columns]
        missing_features = [c for c in feature_columns if c not in df.columns]
        if missing_features:
            formatted_missing = "\n".join(
                f"  - MISSING FEATURE: {feature}" for feature in missing_features
            )
            logger.warning(
                "\n"
                + "!" * 90
                + "\n"
                + "!!! CRITICAL WARNING: EXPECTED FEATURE COLUMNS WERE NOT FOUND IN DATA !!!\n"
                + "This will likely degrade model quality and should be investigated immediately.\n"
                + formatted_missing
                + "\n"
                + "!" * 90
            )
        X = df[available]
    else:
        metadata_columns = feat_eng_config.metadata_columns
        intermediate_columns = feat_eng_config.intermediate_columns
        exclude_columns = feat_eng_config.exclude_columns
        columns_to_exclude = (
            metadata_columns + intermediate_columns + [target_column] + (exclude_columns or [])
        )
        X = df.drop(columns=[col for col in columns_to_exclude if col in df.columns])

    logger.info(f"Prepared {len(X.columns)} features for model prediction")
    logger.debug(f"Feature columns: {list(X.columns)}")

    return X
