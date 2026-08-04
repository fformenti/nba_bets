"""Build the train/validation/test split for an experiment config.

This is the single definition of "which games are in which split". Both the
sklearn training path (``src/ml/training/experiment.py``) and the LLM dataset
builder (``src/ml/llm/dataset.py``) call :func:`build_splits`, so the two model
families are guaranteed to train, validate and test on the same games. The only
difference between them is the encoding: sklearn consumes ``X_*`` directly, the
LLM serializes those same rows to text via ``src/ml/llm/serialization.py``.

Extracted verbatim from ``train_single_model`` so that guarantee holds by
construction rather than by convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

import pandas as pd

from src.config.paths import (
    HOLDOUT_TEST_METADATA_PATH,
    REGULAR_SEASON_GAMES_FEATURES_PATH,
)
from src.ml.config.schema import ExperimentConfig
from src.ml.datasets.splitters import fixed_holdout_split, temporal_split
from src.ml.features.engineering import (
    apply_conference_features,
    create_delta_features,
    resolve_feature_columns,
)
from src.ml.training.data_prep import load_and_validate_data, prepare_data
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Splits:
    """Everything downstream needs from a split, index-aligned throughout.

    ``metadata`` spans every row that survived filtering, so
    ``metadata.loc[X_train.index, "gameId"]`` gives the gameIds of the training
    split — which is how the LLM dataset builder mirrors these splits.
    """

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

    # Two hand-built columns on the untouched feature frame, used by the
    # baseline models that predict from raw record/point-differential deltas.
    X_test_baseline: pd.DataFrame
    y_test_baseline: pd.Series

    metadata: pd.DataFrame
    metadata_test: pd.DataFrame
    games_enriched: pd.DataFrame
    # Target over every surviving row, before splitting — the source of truth
    # for the full class set.
    y: pd.Series
    used_fixed_holdout: bool
    date_range_name: Optional[str]

    def game_ids(self, split: str) -> pd.Series:
        """gameIds for ``split`` in {'train', 'validation', 'test'}."""
        index = {
            "train": self.X_train.index,
            "validation": self.X_val.index,
            "test": self.X_test.index,
        }[split]
        return self.metadata.loc[index, "gameId"]


def build_splits(
    config: ExperimentConfig,
    holdout_path: Optional[Path] = None,
) -> Splits:
    """Load the feature table and split it according to ``config``.

    Steps, in order: load and validate → filter by ``min_season`` → filter by
    conference matchup → drop NA and peel off metadata → resolve the frozen
    holdout → build baseline test frames → engineer delta and conference
    features → select feature columns → split → apply the minimum-games-played
    masks.

    Parameters
    ----------
    config : ExperimentConfig
        Drives every filter, feature and split decision.
    holdout_path : Path, optional
        Frozen holdout metadata. Defaults to
        :data:`src.config.paths.HOLDOUT_TEST_METADATA_PATH`; overridable so
        tests can pin a holdout without depending on local data. When the file
        is absent, falls back to a temporal split.
    """
    if holdout_path is None:
        holdout_path = HOLDOUT_TEST_METADATA_PATH

    data_config = config.data
    split_config = config.splitting
    feat_eng_config = config.feature_engineering
    filters_config = config.filters

    data_path = (
        Path(data_config.path) if data_config.path else Path(REGULAR_SEASON_GAMES_FEATURES_PATH)
    )
    target_column = data_config.target_column
    date_column = data_config.date_column

    metadata_columns = feat_eng_config.metadata_columns
    intermediate_columns = feat_eng_config.intermediate_columns

    minimum_games_train = filters_config.minimum_games_train
    minimum_games_test = filters_config.minimum_games_test
    conference_filter = filters_config.conference_filter
    min_season = filters_config.min_season

    test_size = split_config.test_size
    val_size = split_config.val_size

    if conference_filter not in ["same", "different", "all"]:
        raise ValueError(
            f"conference_filter must be 'same', 'different', or 'all', got '{conference_filter}'"
        )

    games_enriched = load_and_validate_data(
        data_path=data_path, target_column=target_column, date_column=date_column
    )

    if min_season is not None:
        if "season" not in games_enriched.columns:
            raise ValueError("DataFrame must contain 'season' column for min_season filtering")
        games_enriched = games_enriched[games_enriched["season"] >= min_season].copy()
        logger.info(f"Filtered to {len(games_enriched)} games (min_season: {min_season})")

    if conference_filter == "different":
        games_enriched = games_enriched[
            games_enriched["hometeamConference"] != games_enriched["awayteamConference"]
        ].copy()
        logger.info(f"Filtered to {len(games_enriched)} games (different conferences)")
    elif conference_filter == "same":
        games_enriched = games_enriched[
            games_enriched["hometeamConference"] == games_enriched["awayteamConference"]
        ].copy()
        logger.info(f"Filtered to {len(games_enriched)} games (same conference)")
    else:
        logger.info(f"Using all {len(games_enriched)} games (no conference filter)")

    date_range_name = None
    if date_column and date_column in games_enriched.columns:
        dates = pd.to_datetime(games_enriched[date_column], errors="coerce").dropna()
        if not dates.empty:
            date_range_name = (
                f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
            )

    df, y, metadata = prepare_data(
        df=games_enriched,
        target_column=target_column,
        drop_na=data_config.drop_na,
        metadata_columns=metadata_columns,
    )

    # Resolve holdout indices once — used for both baseline and main splits
    used_fixed_holdout = holdout_path.exists()
    holdout_indices = None
    if used_fixed_holdout:
        holdout_meta = pd.read_csv(holdout_path, usecols=["gameId"])
        holdout_game_ids = set(holdout_meta["gameId"].astype(int))
        holdout_indices = metadata[metadata["gameId"].isin(holdout_game_ids)].index
        logger.info(
            f"Using fixed holdout: {len(holdout_indices)} games "
            f"({len(holdout_game_ids)} in file, {len(holdout_indices)} matched)"
        )
    else:
        logger.warning(
            f"Fixed holdout not found at {holdout_path}. "
            "Falling back to temporal split — test set will shift as new games are added. "
            "Run `make build-holdout-set` to freeze the holdout."
        )

    # Special Dataframe for baseline models
    if used_fixed_holdout:
        X_test_baseline = df.loc[holdout_indices].copy()
        y_test_baseline = y.loc[holdout_indices].copy()
    else:
        _, _, X_test_baseline, _, _, y_test_baseline = temporal_split(
            df, y, date_column=date_column, test_size=test_size, val_size=val_size
        )
    X_test_baseline["record_L82_delta"] = (
        X_test_baseline["record_L82_HT"] - X_test_baseline["record_L82_VT"]
    )
    X_test_baseline["pts_diff_avg_L82_delta"] = (
        X_test_baseline["pts_diff_avg_L82_HT"] - X_test_baseline["pts_diff_avg_L82_VT"]
    )

    df = create_delta_features(df, feat_eng_config.features)
    df = apply_conference_features(df, conference_filter)

    if feat_eng_config.selection_mode == "inclusion":
        feature_columns = resolve_feature_columns(
            feat_eng_config.features,
            conference_filter,
            momentum_pairs=config.feature_engineering.momentum_pairs or None,
        )
        keep_cols = feature_columns + [date_column]
        available = [c for c in keep_cols if c in df.columns]
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
        exclude_cols = metadata_columns + intermediate_columns + [target_column]
        cols_to_drop = [col for col in exclude_cols if col in df.columns]
        X = df.drop(columns=cols_to_drop)

    if date_column not in X.columns:
        raise ValueError(f"Date column '{date_column}' required for temporal split")
    if used_fixed_holdout:
        X_train, X_val, X_test, y_train, y_val, y_test = fixed_holdout_split(
            X, y, holdout_indices=holdout_indices, date_column=date_column, val_size=val_size
        )
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = temporal_split(
            X, y, date_column=date_column, test_size=test_size, val_size=val_size
        )
    X_train = cast(pd.DataFrame, X_train.drop(columns=[date_column]))
    X_val = X_val.drop(columns=[date_column])
    X_test = X_test.drop(columns=[date_column])

    train_gp = games_enriched.loc[X_train.index]
    train_mask = (train_gp["games_played_HT"] > minimum_games_train) & (
        train_gp["games_played_VT"] > minimum_games_train
    )
    X_train = X_train.loc[train_mask]
    y_train = cast(pd.Series, y_train.loc[train_mask])

    val_gp = games_enriched.loc[X_val.index]
    val_mask = (val_gp["games_played_HT"] > minimum_games_train) & (
        val_gp["games_played_VT"] > minimum_games_train
    )
    X_val = cast(pd.DataFrame, X_val.loc[val_mask])
    y_val = cast(pd.Series, y_val.loc[val_mask])

    test_gp = games_enriched.loc[X_test.index]
    test_mask = (test_gp["games_played_HT"] > minimum_games_test) & (
        test_gp["games_played_VT"] > minimum_games_test
    )
    X_test = X_test.loc[test_mask]
    y_test = cast(pd.Series, y_test.loc[test_mask])

    X_test_baseline = X_test_baseline.loc[X_test.index]
    y_test_baseline = y_test_baseline.loc[X_test.index]

    metadata_test = metadata.loc[X_test.index].copy()

    return Splits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        X_test_baseline=X_test_baseline,
        y_test_baseline=y_test_baseline,
        metadata=metadata,
        metadata_test=metadata_test,
        games_enriched=games_enriched,
        y=y,
        used_fixed_holdout=used_fixed_holdout,
        date_range_name=date_range_name,
    )
