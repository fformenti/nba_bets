"""I/O helpers for prediction pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd


UPCOMING_RENAMES = {
    "homeTeamName": "hometeamName",
    "homeTeamCity": "hometeamPrename",
    "awayTeamName": "awayteamName",
    "awayTeamCity": "awayteamPrename",
}

REQUIRED_UPCOMING_COLUMNS = {
    "gameId",
    "gameDate",
    "hometeamId",
    "awayteamId",
}


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_upcoming_games(
    input_path: Path, max_files: Optional[int] = None
) -> pd.DataFrame:
    """
    Load upcoming games from a JSON file or directory of JSON files.
    """
    input_path = Path(input_path)
    if input_path.is_dir():
        files = sorted(input_path.glob("*.json"))
        if max_files is not None:
            files = files[:max_files]
        payloads = [_load_json(path) for path in files]
    elif input_path.is_file():
        payloads = [_load_json(input_path)]
    else:
        raise FileNotFoundError(f"Upcoming games path not found: {input_path}")

    if not payloads:
        return pd.DataFrame()

    df = pd.DataFrame(payloads)
    df = df.rename(columns=UPCOMING_RENAMES)

    missing = REQUIRED_UPCOMING_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Upcoming games missing required columns: {missing}")

    df["gameId"] = pd.to_numeric(df["gameId"], errors="coerce").astype("Int64")
    df["hometeamId"] = pd.to_numeric(df["hometeamId"], errors="coerce").astype("Int64")
    df["awayteamId"] = pd.to_numeric(df["awayteamId"], errors="coerce").astype("Int64")
    df["gameDate"] = pd.to_datetime(df["gameDate"], errors="coerce")

    df = df.dropna(subset=["gameId", "hometeamId", "awayteamId", "gameDate"]).copy()
    return df.reset_index(drop=True)


def load_historical_features(features_path: Path) -> pd.DataFrame:
    """
    Load historical features used to build inference rows.
    """
    features_path = Path(features_path)
    if not features_path.exists():
        raise FileNotFoundError(f"Historical features not found: {features_path}")

    df = pd.read_csv(features_path, parse_dates=["gameDate"])
    if df.empty:
        raise ValueError(f"Historical features file is empty: {features_path}")
    return df
