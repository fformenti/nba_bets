"""Feature construction for prediction pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.ml.config import ExperimentConfig
from src.ml.features import (
    create_conference_delta,
    create_delta_features,
    get_home_conference_vs_away_conference_record,
)


GLOBAL_FEATURE_COLUMNS = [
    "east_record_adjusted",
    "west_record_adjusted",
    "games_played_east_vs_west",
    "east_record_at_east",
    "games_played_at_east",
    "games_played_at_west",
    "west_record_at_west",
    "east_wins_pct_L1",
]

HOME_SUFFIXES = ("_HT", "_HT_at_home")
AWAY_SUFFIXES = ("_VT", "_VT_on_road")


@dataclass
class FeatureBuildResult:
    features: pd.DataFrame
    metadata: pd.DataFrame


def _select_feature_columns(columns: list[str], suffixes: tuple[str, ...]) -> list[str]:
    return [col for col in columns if col.endswith(suffixes)]


def _prepare_rest_context(historical: pd.DataFrame) -> pd.DataFrame:
    home = historical[
        ["gameDate", "hometeamId", "rested_days_HT", "days_at_home"]
    ].rename(
        columns={
            "hometeamId": "teamId",
            "rested_days_HT": "rested_days",
            "days_at_home": "days_at_location",
        }
    )
    home["last_location"] = "home"

    away = historical[
        ["gameDate", "awayteamId", "rested_days_VT", "days_on_road"]
    ].rename(
        columns={
            "awayteamId": "teamId",
            "rested_days_VT": "rested_days",
            "days_on_road": "days_at_location",
        }
    )
    away["last_location"] = "away"

    combined = pd.concat([home, away], ignore_index=True)
    combined = combined.sort_values("gameDate")
    return combined.groupby("teamId", as_index=False).tail(1)


def _apply_rest_adjustments(
    df: pd.DataFrame, last_games: pd.DataFrame, location: str
) -> pd.DataFrame:
    team_col = "hometeamId" if location == "home" else "awayteamId"
    rested_col = "rested_days_HT" if location == "home" else "rested_days_VT"
    days_col = "days_at_home" if location == "home" else "days_on_road"

    merged = df.merge(
        last_games,
        how="left",
        left_on=team_col,
        right_on="teamId",
        suffixes=("", "_last"),
    )
    if merged.columns.duplicated().any():
        merged = merged.loc[:, ~merged.columns.duplicated()].copy()
    delta_days = (merged["gameDate"] - merged["gameDate_last"]).dt.days
    delta_days = delta_days.where(delta_days >= 0)

    rest_days = np.maximum(delta_days - 1, 0)
    same_location = merged["last_location"] == location
    days_at_location = np.where(
        same_location,
        merged["days_at_location"] + delta_days,
        delta_days,
    )

    merged[rested_col] = np.where(delta_days.notna(), rest_days, merged[rested_col])
    merged[days_col] = np.where(delta_days.notna(), days_at_location, merged[days_col])

    merged = merged.drop(
        columns=[
            "teamId",
            "gameDate_last",
            "rested_days",
            "days_at_location",
            "last_location",
        ]
    )
    return merged


def build_features_for_upcoming(
    upcoming: pd.DataFrame,
    historical: pd.DataFrame,
    config: ExperimentConfig,
    conference_filter: str = "different",
    add_conference_delta: bool = False,
) -> FeatureBuildResult:
    """
    Build model-ready features for upcoming games using historical feature snapshots.
    """
    if upcoming.empty:
        return FeatureBuildResult(features=pd.DataFrame(), metadata=pd.DataFrame())

    df = upcoming.copy()
    df["gameDate"] = pd.to_datetime(df["gameDate"], errors="coerce")
    df["gameDateOnlyStr"] = df["gameDate"].dt.strftime("%Y-%m-%d")
    df["hometeamId"] = pd.to_numeric(df["hometeamId"], errors="coerce").astype("int64")
    df["awayteamId"] = pd.to_numeric(df["awayteamId"], errors="coerce").astype("int64")

    historical = historical.copy()
    historical["gameDate"] = pd.to_datetime(historical["gameDate"], errors="coerce")
    if "hometeamId" in historical.columns:
        historical["hometeamId"] = pd.to_numeric(
            historical["hometeamId"], errors="coerce"
        ).astype("int64")
    if "awayteamId" in historical.columns:
        historical["awayteamId"] = pd.to_numeric(
            historical["awayteamId"], errors="coerce"
        ).astype("int64")

    history_columns = historical.columns.tolist()
    home_cols = _select_feature_columns(history_columns, HOME_SUFFIXES)
    away_cols = _select_feature_columns(history_columns, AWAY_SUFFIXES)

    home_cols += [
        col
        for col in ["rested_days_HT", "days_at_home", "hometeamConference"]
        if col in history_columns
    ]
    away_cols += [
        col
        for col in ["rested_days_VT", "days_on_road", "awayteamConference"]
        if col in history_columns
    ]
    global_cols = [col for col in GLOBAL_FEATURE_COLUMNS if col in history_columns]

    home_history = historical[["gameDate", "hometeamId"] + home_cols].sort_values(
        "gameDate"
    )
    df = pd.merge_asof(
        df.sort_values("gameDate"),
        home_history,
        on="gameDate",
        by="hometeamId",
        direction="backward",
        allow_exact_matches=False,
    )

    away_history = historical[["gameDate", "awayteamId"] + away_cols].sort_values(
        "gameDate"
    )
    df = pd.merge_asof(
        df.sort_values("gameDate"),
        away_history,
        on="gameDate",
        by="awayteamId",
        direction="backward",
        allow_exact_matches=False,
    )

    global_history = (
        historical[["gameDate"] + global_cols]
        .drop_duplicates(subset=["gameDate"])
        .sort_values("gameDate")
    )
    df = pd.merge_asof(
        df.sort_values("gameDate"),
        global_history,
        on="gameDate",
        direction="backward",
        allow_exact_matches=False,
    )

    last_games = _prepare_rest_context(historical)
    df = _apply_rest_adjustments(df, last_games, location="home")
    df = _apply_rest_adjustments(df, last_games, location="away")

    if conference_filter == "different":
        df = df[df["hometeamConference"] != df["awayteamConference"]].copy()
    elif conference_filter == "same":
        df = df[df["hometeamConference"] == df["awayteamConference"]].copy()

    delta_df = create_delta_features(
        df.copy(),
        lags=config.feature_engineering.lags,
        location_lags=config.feature_engineering.location_lags,
    )
    delta_cols = [
        col
        for col in delta_df.columns
        if col.endswith("_delta") or col.endswith("_delta_at_location")
    ]
    for col in delta_cols:
        df[col] = delta_df[col]

    if conference_filter == "different":
        conf_df = get_home_conference_vs_away_conference_record(df.copy())
        for col in [
            "home_conference_vs_away_conference_record",
            "games_played_at_home_conference",
        ]:
            if col in conf_df.columns:
                df[col] = conf_df[col]
    if add_conference_delta:
        conf_delta = create_conference_delta(df.copy())
        if "conference_diff_home_advantage_pct" in conf_delta.columns:
            df["conference_diff_home_advantage_pct"] = conf_delta[
                "conference_diff_home_advantage_pct"
            ]
            df["conference_diff_east_pct"] = conf_delta[
                "conference_diff_home_advantage_pct"
            ]

    metadata_cols = [
        col
        for col in [
            "gameId",
            "gameDate",
            "hometeamId",
            "hometeamName",
            "hometeamCity",
            "awayteamId",
            "awayteamName",
            "awayteamCity",
            "arenaName",
            "arenaCity",
        ]
        if col in df.columns
    ]
    metadata = df[metadata_cols].copy()
    return FeatureBuildResult(features=df, metadata=metadata)
