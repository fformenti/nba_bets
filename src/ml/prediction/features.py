"""Feature construction for upcoming-game prediction.

Mirrors the training-time feature path: the same ``create_features_tables`` /
``merge_features`` from the ETL layer, then the same delta and conference
feature engineering. Upcoming games are concatenated onto history first, because
rolling features (records, point differentials, rest) are only defined relative
to the games that came before.
"""

from __future__ import annotations


import pandas as pd

from src.config.paths import DEFAULT_FEATURES_CONFIG_PATH, PREDICTION_FEATURES_DIR
from src.etl.features.aggregator import (
    create_features_tables_from_config,
    merge_features,
)
from src.etl.utils.common import enrich_games_locations, get_nba_season
from src.ml.config.loader import load_features_config
from src.ml.config.schema import ExperimentConfig
from src.ml.features.engineering import (
    create_conference_features,
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

    # Set placeholder values for upcoming games.
    #
    # `winner = 0` marks the row as not yet played — see
    # src.etl.utils.common.has_result, which is how the builders that accumulate
    # across games know to leave these rows out of everyone else's totals.
    df["postponed"] = 0
    df["overtimes"] = None
    df["winner"] = 0
    df["homeScore"] = 0
    df["awayScore"] = 0
    df["arenaId"] = None
    df["attendance"] = None
    df["gameType"] = "Regular Season"

    # `win_bool` and `pts_diff` exist on the ETL's historical frame, and
    # join_games_and_teams_feature only suffixes columns the games frame does not
    # already have. Without them here the prediction path grew a set of
    # win_bool_HT / pts_diff_VT / ... columns the training path never had — the
    # same silent divergence between the two feature paths that let the distance
    # bug below go unnoticed.
    df["win_bool"] = 0
    df["pts_diff"] = 0

    # Likewise `neutral_court`, which the ETL adds in make_features. Deriving it
    # here rather than calling add_neutral_court keeps the prediction path free
    # of a dependency on teams_arena.csv and gives the identical answer: that
    # function falls back to the label flag whenever arenaId is null, and an
    # upcoming game has no arenaId.
    if "is_neutral_court_game" in df.columns:
        df["neutral_court"] = df["is_neutral_court_game"].fillna(False).astype(int)
    else:
        df["neutral_court"] = 0

    # Travel distance is computed from where each team last played, so the slate
    # needs its locations like any other game. Only the ETL's ingestion path
    # enriched them, which left every upcoming game with NaN locations and, via
    # the unknown-location branch of the distance builder, a travel distance of
    # zero for the away team of every game ever predicted. In training that same
    # feature carries the real mileage.
    df = enrich_games_locations(df)

    return df


def build_prediction_feature_base(
    upcoming_games: pd.DataFrame,
    historical_features: pd.DataFrame,
    features_config=None,
) -> pd.DataFrame:
    """
    Build the ETL half of the prediction features: one slate, one set of tables.

    This is the run-level work — it depends only on the slate and the history,
    never on which model the rows are headed for. Call it once per run and hand
    the result to ``apply_model_feature_engineering`` for each model.

    The tables are written to ``PREDICTION_FEATURES_DIR``, not to the ETL's
    ``REGULAR_SEASON_FEATURES_DIR``: the frame here is a single season plus the
    slate, and writing that under the ETL's filenames left `make build-features`
    output truncated after every prediction run.

    Parameters
    ----------
    upcoming_games : pd.DataFrame
        Prepared upcoming games DataFrame
    historical_features : pd.DataFrame
        Historical features DataFrame
    features_config : optional
        ETL features config. Defaults to ``configs/features.yaml`` — the same
        file `make build-features` uses.

    Returns
    -------
    pd.DataFrame
        Upcoming games with every merged feature column, before any
        model-specific engineering.
    """
    logger.info("Building features for prediction")

    if features_config is None:
        features_config = load_features_config(DEFAULT_FEATURES_CONFIG_PATH)

    # Rolling features are defined relative to prior games, so upcoming rows
    # have to sit on top of history to be computable at all.
    #
    # A game in the upcoming slate may already have been played and ingested
    # into history: the daily loop writes the slate before tip-off and does not
    # clear it afterwards, so `fetch-upcoming-games` and `append-game-results`
    # overlap by design. History wins — it carries the real score, whereas the
    # upcoming row is a 0-0 placeholder from `fix_upcoming_games_cols`.
    #
    # Keeping both is not merely inaccurate, it is fatal: the duplicate gameId
    # propagates into every feature table, and `merge_features` joins on
    # (gameId, season, teamId) roughly two dozen times, doubling the frame at
    # each join. A 10-game slate with 7 already-played games reached 7.3M rows
    # by the 20th join and the process was OOM-killed.
    historical_combined = pd.concat([historical_features, upcoming_games])
    already_played = historical_combined["gameId"].duplicated()
    if already_played.any():
        logger.info(
            f"{int(already_played.sum())} upcoming game(s) already present in history; "
            "keeping the historical row: "
            f"{sorted(historical_combined.loc[already_played, 'gameId'].tolist())}"
        )
        historical_combined = historical_combined[~already_played]

    logger.info("Creating feature tables")
    create_features_tables_from_config(
        historical_combined, features_config, output_dir=PREDICTION_FEATURES_DIR
    )

    # Merge features for upcoming games
    upcoming_with_features = merge_features(
        upcoming_games, features_dir=PREDICTION_FEATURES_DIR
    )

    # Drop columns that are not needed for feature engineering
    # These columns are added for merge_features but shouldn't be in final features
    columns_to_drop_after_merge = ["winner", "winnerteamConference"]
    upcoming_with_features = upcoming_with_features.drop(
        columns=[
            col for col in columns_to_drop_after_merge if col in upcoming_with_features.columns
        ]
    )

    return upcoming_with_features


def apply_model_feature_engineering(
    base_features: pd.DataFrame,
    experiment_config: ExperimentConfig,
) -> pd.DataFrame:
    """
    Build the model half of the prediction features, for one model.

    Two configs, two jobs. The **ETL** features config decided which feature
    *tables* got built, back in ``build_prediction_feature_base``. The
    **experiment** config here decides which of those columns this model
    consumes (delta features, conference features, selection).

    Conflating them was a bug: an experiment may set ``record.lags: []`` while
    leaving a derived group like ``sos_adj_record`` enabled, which is coherent
    for feature selection but incoherent as an ETL instruction, and blew up as
    ``KeyError: 'record_L5'`` inside the SOS-adjusted record builder.

    Parameters
    ----------
    base_features : pd.DataFrame
        Output of ``build_prediction_feature_base``. Not mutated — each model
        adds its own columns, so they must not see each other's.
    experiment_config : ExperimentConfig
        Experiment configuration for the model these rows are headed for

    Returns
    -------
    pd.DataFrame
        DataFrame with features ready for prediction
    """
    feat_eng_config = experiment_config.feature_engineering

    # Same two steps as the training path, in the same order — see
    # src/ml/datasets/splits.py::build_splits.
    logger.info("Creating delta features")
    upcoming_with_features = create_delta_features(
        base_features.copy(),
        feat_eng_config.features,
    )
    upcoming_with_features = create_conference_features(upcoming_with_features)

    return upcoming_with_features


def build_features_for_prediction(
    upcoming_games: pd.DataFrame,
    historical_features: pd.DataFrame,
    experiment_config: ExperimentConfig,
    features_config=None,
) -> pd.DataFrame:
    """Build prediction features end to end, for a single model.

    Both halves in one call. The pipeline calls them separately so the ETL
    tables are built once per slate rather than once per caller.
    """
    base_features = build_prediction_feature_base(
        upcoming_games=upcoming_games,
        historical_features=historical_features,
        features_config=features_config,
    )
    return apply_model_feature_engineering(
        base_features=base_features,
        experiment_config=experiment_config,
    )


def prepare_features_for_model(
    df: pd.DataFrame,
    experiment_config: ExperimentConfig,
    target_column: str,
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

    Returns
    -------
    pd.DataFrame
        Features DataFrame ready for model prediction
    """
    feat_eng_config = experiment_config.feature_engineering

    if feat_eng_config.selection_mode == "inclusion":
        feature_columns = resolve_feature_columns(feat_eng_config.features)
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
