"""Feature engineering utilities for NBA games data."""

import logging
from typing import List, Optional

import numpy as np
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
    "margin": lambda lags: [f"margin_avg_L{lag}" for lag in lags],
    "margin_scaled": lambda lags: [f"margin_scaled_avg_L{lag}" for lag in lags],
    "sos": lambda lags: [f"sos_L{lag}" for lag in lags],
    "sos_adj_record": lambda lags: [f"sos_adj_record_L{lag}" for lag in lags],
    "gds": lambda lags: [f"gds_L{lag}" for lag in lags],
    "distance": lambda lags: [f"distance_L{lag}" for lag in lags],
    "rested_days": lambda _: ["rested_days"],
    "back_to_back": lambda _: ["back_to_back"],
    "streak": lambda _: ["streak"],
    "last_season_record": lambda _: ["last_season_record"],
    "adjusted_last_season_record": lambda _: ["adjusted_last_season_record"],
    # compute_playoff_flags emits several columns per team, and two of them are
    # precursors rather than groups. conf_rank feeds the games-behind/-ahead
    # formulas and the clinch logic and is never a training candidate, so it
    # lives in src/config/constants.py::INTERMEDIATE_COLUMNS. The three clinch
    # booleans were three independent groups from 2026-08-12 until 2026-08-18,
    # when they collapsed into the single ordinal winning_incentive below —
    # they are still computed by the ETL, but they reach a model only through
    # create_winning_incentive_features, not as configurable groups.
    "games_behind_leader": lambda _: ["games_behind_leader"],
    "games_behind_above": lambda _: ["games_behind_above"],
    "games_ahead_of_below": lambda _: ["games_ahead_of_below"],
    "winning_incentive": lambda _: ["winning_incentive"],
}

# Groups measuring a lower-is-better quantity, so the operands swap to
# ``VT - HT``. Every delta in the pipeline is oriented so that a *positive*
# value favours the home team; for distance travelled and back-to-backs, more
# is worse, so the plain ``HT - VT`` would point the other way. That convention
# is not cosmetic — src/ml/llm/serialization.py::SYSTEM_PROMPT states it to the
# model verbatim, and a SHAP sign is read against it.
INVERTED_DELTA_GROUPS = {"distance", "back_to_back"}

# Groups that have HT_at_home / VT_on_road location variants
LOCATION_VARIANT_GROUPS = {
    "record",
    "margin",
    "margin_scaled",
    "last_season_record",
    "adjusted_last_season_record",
    "sos_adj_record",
    "gds",
}


def _numeric(series: pd.Series) -> pd.Series:
    """Cast a boolean column to int so arithmetic on it is defined.

    The three clinch flags are written as booleans by the aggregator and read
    back as booleans from the CSV; numpy refuses ``-`` on bool. That bites in
    two places: ``create_winning_incentive_features`` keys on their integer
    triple, and any future boolean group would take a delta rather than a
    TypeError.
    """
    return series.astype(int) if series.dtype == bool else series


# Label → ordinal for winning_incentive. The two "fighting" states deliberately
# share a value: a team still chasing a berth and a team already in but still
# chasing a seed are both playing their starters, which is the mechanism this
# feature exists to capture. Edit the numbers here and nowhere else.
WINNING_INCENTIVE_VALUES = {
    "fighting": 1,
    "fighting for seed": 1,
    "resting for playoffs": 0,
    "eliminated": -1,
}

# (eliminated_from_playoffs, clinched_playoff_berth, clinched_final_seed) → label.
# Only four of the eight combinations are reachable, and that is a property of
# the builder rather than of the data: src/etl/features/playoff_standings.py
# ::_rank_conference_snapshot evaluates clinched_final_seed only when
# clinched_playoff_berth is true, and a team that is eliminated has clinched
# neither. Anything else means the builder changed — see the raise below.
_WINNING_INCENTIVE_LABELS = {
    (0, 0, 0): "fighting",
    (1, 0, 0): "eliminated",
    (0, 1, 0): "fighting for seed",
    (0, 1, 1): "resting for playoffs",
}

_WINNING_INCENTIVE_SOURCES = (
    "eliminated_from_playoffs",
    "clinched_playoff_berth",
    "clinched_final_seed",
)


def create_winning_incentive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the three clinch booleans into one ordinal, per team.

    ``eliminated_from_playoffs``, ``clinched_playoff_berth`` and
    ``clinched_final_seed`` are not three independent facts — they are one
    ordinal quantity spelled across three columns: how much a team still has to
    play for. Offered separately, all six ``_HT``/``_VT`` columns were rejected
    by Boruta. Offered as one signed number they carry a sign that already
    points the right way, so the group takes a plain ``HT - VT`` delta and is
    *not* in ``INVERTED_DELTA_GROUPS``.

    Emits, per team suffix:

    ``winning_incentive_{HT,VT}``
        -1 eliminated, 0 seed locked and resting, 1 still fighting.
    ``winning_incentive_label_{HT,VT}``
        The same thing in words, for EDA and plotting. Never enters the numeric
        path: no group names it, so ``resolve_feature_columns`` never asks for
        it.

    The six source columns are dropped on the way out. They stopped being
    feature groups when this function landed, so nothing downstream would drop
    them, and a stray boolean pair surviving into an exclusion-mode feature
    frame would silently re-add the very columns this collapse removed.

    An unreachable combination raises rather than defaulting: a silent fallback
    would turn a broken clinch builder into a plausible-looking feature.
    """
    df = df.copy()
    columns_to_drop: list[str] = []

    for suffix in (HOME_SUFFIX, AWAY_SUFFIX):
        source_cols = [f"{name}_{suffix}" for name in _WINNING_INCENTIVE_SOURCES]
        missing = [col for col in source_cols if col not in df.columns]
        if missing:
            logger.warning(
                f"Skipping winning_incentive_{suffix}: missing {', '.join(missing)}"
            )
            continue

        combos = pd.DataFrame(
            {col: _numeric(df[col]).astype("Int64") for col in source_cols}
        )
        keys = pd.Series(
            list(zip(*(combos[col] for col in source_cols))), index=df.index
        )
        labels = keys.map(_WINNING_INCENTIVE_LABELS)

        unmapped = keys[labels.isna()]
        if not unmapped.empty:
            counts = unmapped.value_counts()
            detail = ", ".join(
                f"({', '.join('NA' if pd.isna(v) else str(int(v)) for v in combo)})"
                f": {n} row(s)"
                for combo, n in counts.items()
            )
            raise ValueError(
                f"Unreachable playoff-flag combination in {suffix} columns "
                f"(eliminated_from_playoffs, clinched_playoff_berth, "
                f"clinched_final_seed) — {detail}. Only "
                f"{sorted(_WINNING_INCENTIVE_LABELS)} are producible by "
                "compute_playoff_flags; a new combination means the clinch "
                "logic changed and this mapping has to change with it."
            )

        df[f"winning_incentive_label_{suffix}"] = labels
        df[f"winning_incentive_{suffix}"] = labels.map(WINNING_INCENTIVE_VALUES).astype(
            int
        )
        columns_to_drop.extend(source_cols)

    if columns_to_drop:
        df.drop(columns=columns_to_drop, inplace=True)
        logger.info(
            f"Created winning_incentive from {len(columns_to_drop)} clinch columns"
        )
    return df


def create_delta_features(
    df: pd.DataFrame,
    features_config: FeaturesMapConfig,
) -> pd.DataFrame:
    """
    Create delta features from the per-team columns, per feature group config.

    For each enabled feature group with ``delta=True``, creates
    ``{prefix}_delta = {prefix}_HT - {prefix}_VT`` — except for the groups in
    ``INVERTED_DELTA_GROUPS``, where the operands swap so that a positive delta
    still favours the home team. Source HT/VT columns are always dropped after
    delta creation.

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
                columns_to_drop.extend(
                    [f"{prefix}_{HOME_SUFFIX}", f"{prefix}_{AWAY_SUFFIX}"]
                )
            if group_name in LOCATION_VARIANT_GROUPS:
                loc_prefixes = prefix_fn(group_config.location_lags)
                for prefix in loc_prefixes:
                    columns_to_drop.extend(
                        [
                            f"{prefix}_{HOME_AT_HOME_SUFFIX}",
                            f"{prefix}_{AWAY_ON_ROAD_SUFFIX}",
                        ]
                    )
            continue

        if group_config.delta:
            for prefix in prefixes:
                home_col = f"{prefix}_{HOME_SUFFIX}"
                away_col = f"{prefix}_{AWAY_SUFFIX}"
                if home_col not in df.columns or away_col not in df.columns:
                    logger.debug(f"Skipping delta for {prefix}: columns not found")
                    continue
                home, away = _numeric(df[home_col]), _numeric(df[away_col])
                if group_name in INVERTED_DELTA_GROUPS:
                    home, away = away, home
                df[f"{prefix}_delta"] = home - away
                created_features.append(f"{prefix}_delta")
                columns_to_drop.extend([home_col, away_col])

            if group_name in LOCATION_VARIANT_GROUPS:
                loc_prefixes = prefix_fn(group_config.location_lags)
                for prefix in loc_prefixes:
                    home_col = f"{prefix}_{HOME_AT_HOME_SUFFIX}"
                    away_col = f"{prefix}_{AWAY_ON_ROAD_SUFFIX}"
                    if home_col not in df.columns or away_col not in df.columns:
                        logger.debug(
                            f"Skipping location delta for {prefix}: columns not found"
                        )
                        continue
                    home, away = _numeric(df[home_col]), _numeric(df[away_col])
                    if group_name in INVERTED_DELTA_GROUPS:
                        home, away = away, home
                    df[f"{prefix}_at_location_delta"] = home - away
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

    # neutral_court: game-level scalar — no HT/VT split, no delta
    if not features_config.neutral_court.enabled:
        columns_to_drop.append("neutral_court")

    # home_adv: league-wide scalar — no HT/VT split, no delta
    if not features_config.home_adv.enabled:
        columns_to_drop.append("home_adv_L1000")

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
        margin_momentum_L{short}_L{long}_delta = margin_avg_L{short}_delta - margin_avg_L{long}_delta

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


CONFERENCE_FEATURES = [
    "conference_diff_home_advantage_pct",
    "conference_home_court_advantage_pct",
]

# Columns create_conference_features reads and then removes from the frame.
# Nothing downstream may consume them: they are conference-season-level, so on a
# same-conference game they describe a contest that isn't happening.
_CONFERENCE_INTERMEDIATES = [
    "hometeamConference",
    "awayteamConference",
    "east_record_adjusted",
    "west_record_adjusted",
    "east_record_at_east",
    "west_record_at_west",
    "games_played_east_vs_west",
    "last_season_east_record_adjusted",
    "last_season_west_record_adjusted",
    "last_season_east_record_at_east",
    "last_season_west_record_at_west",
]


def create_conference_features(
    df: pd.DataFrame, saturation_games: int = 20
) -> pd.DataFrame:
    """Both conference features, defined for every game.

    Each measures a conference-vs-conference advantage — a quantity that exists
    only for interconference games. On a same-conference game both teams are
    drawn from the same pool, so the effect doesn't cancel *approximately*, it
    cancels **exactly**; the neutral value is a known truth rather than an
    imputation guess. Both features are therefore 0.0 there:

    ``conference_diff_home_advantage_pct``
        The venue-balanced East/West gap, signed by who is hosting.
        ``east_record_adjusted`` averages East's win rate hosting and visiting,
        so venue is deliberately averaged out. ``conference_diff`` is +1 when
        the East hosts, -1 when the West hosts, and 0 within a conference —
        which is what zeroes the product.

    ``conference_home_court_advantage_pct``
        Exactly the venue component the average above discards: how well the
        home team's conference holds its own floor against the other one,
        centered so that parity is 0.0. Positive means the home team belongs to
        the conference that defends home court better in interconference play.

    Centering matters because the neutral value has to be expressible. A signed
    difference is neutral at 0 and a raw win rate at 0.5; subtracting 0.5 makes
    "no effect" and "no data" the same number, so the linear and neural paths
    don't have to learn where neutral sits, and the LLM prompt renders 0.0
    rather than a spurious .500.

    Leaving the second feature ungated would be a confound, not noise: its
    ingredients are defined on every date, so the raw lookup happily reports
    "the East holds home court at .550" on an all-East game, handing every East
    home team one value and every West home team another.

    Early in a season the current-season inputs (``east_record_adjusted`` etc.)
    are built on a thin interconference sample. Each is blended with last
    season's final record (``last_season_east_record_adjusted`` etc., from
    ``src/etl/features/east_vs_west.py::make_last_season_east_west_record``),
    ramping linearly from last season's value at 0 interconference games played
    to this season's value alone at ``saturation_games`` — the same style of
    ramp as the within-season sample weight in
    ``src/ml/training/weighting.py``, applied to a feature value instead of a
    training weight. A game in the dataset's first season has no prior season
    to blend with, so it uses the current-season value outright.

    Note the intermediates are dropped on the way out — see
    ``_CONFERENCE_INTERMEDIATES``.
    """
    df = df.copy()

    is_cross_conference = df["hometeamConference"] != df["awayteamConference"]

    ramp_weight = (df["games_played_east_vs_west"] / saturation_games).clip(0.0, 1.0)

    def _blend(current_col: str, last_season_col: str) -> pd.Series:
        blended = ramp_weight * df[current_col] + (1 - ramp_weight) * df[last_season_col]
        return blended.where(df[last_season_col].notna(), df[current_col])

    east_record_adjusted = _blend("east_record_adjusted", "last_season_east_record_adjusted")
    west_record_adjusted = _blend("west_record_adjusted", "last_season_west_record_adjusted")
    east_record_at_east = _blend("east_record_at_east", "last_season_east_record_at_east")
    west_record_at_west = _blend("west_record_at_west", "last_season_west_record_at_west")

    # East=1, West=0 → +1 East hosts West, -1 West hosts East, 0 same conference.
    conference_diff = df["hometeamConference"].map({"East": 1, "West": 0}) - df[
        "awayteamConference"
    ].map({"East": 1, "West": 0})

    df["conference_diff_home_advantage_pct"] = conference_diff * (
        east_record_adjusted - west_record_adjusted
    )

    home_conference_home_record = np.where(
        df["hometeamConference"] == "East",
        east_record_at_east,
        west_record_at_west,
    )
    df["conference_home_court_advantage_pct"] = np.where(
        is_cross_conference, home_conference_home_record - 0.5, 0.0
    )

    df = df.drop(
        columns=[col for col in _CONFERENCE_INTERMEDIATES if col in df.columns]
    )

    logger.info(
        f"Created {', '.join(CONFERENCE_FEATURES)} "
        f"({(~is_cross_conference).sum()} same-conference game(s) set to 0.0)"
    )
    return df


def resolve_feature_columns(
    features_config: FeaturesMapConfig,
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

    # neutral_court: game-level scalar — no HT/VT split, no delta
    if features_config.neutral_court.enabled:
        columns.append("neutral_court")

    # home_adv: league-wide scalar — no HT/VT split, no delta
    if features_config.home_adv.enabled:
        columns.append("home_adv_L1000")

    # Conference features (created dynamically by create_conference_features)
    columns.extend(CONFERENCE_FEATURES)

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
