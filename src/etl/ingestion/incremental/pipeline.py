"""Incremental ingestion pipeline for new games data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import (
    PROJECT_ROOT,
    LOCAL_GAMES_FEATURES_PATH,
)
from src.etl.features import create_features_tables, merge_features
from src.etl.ingestion import create_teams_history_table, filter_regular_season_games
from src.etl.ingestion.raw_games import parse_raw_games, get_nba_season
from src.etl.transformation import add_conference
from src.ml.config.loader import load_experiment_config
from src.utils.logging_config import setup_logging, get_logger

from .agents import (
    LLMColumnMappingAgent,
    OpenAIChatClient,
    QualityAssuranceAgent,
    RowNormalizationAgent,
)
from .config import IncrementalIngestionConfig, load_incremental_config
from .io import archive_files, discover_incremental_files, read_incremental_file
from .schema import CANONICAL_GAME_COLUMNS, REQUIRED_GAME_COLUMNS

logger = get_logger(__name__)


def _load_existing_games(raw_games_path: Path) -> pd.DataFrame:
    if not raw_games_path.exists():
        return pd.DataFrame(columns=CANONICAL_GAME_COLUMNS)
    return pd.read_csv(raw_games_path, parse_dates=["gameDate"])


def _dedupe_incremental(
    incremental: pd.DataFrame, existing: pd.DataFrame, dedupe_key: str
) -> pd.DataFrame:
    if dedupe_key in incremental.columns and dedupe_key in existing.columns:
        existing_ids = pd.to_numeric(existing[dedupe_key], errors="coerce")
        incremental_ids = pd.to_numeric(incremental[dedupe_key], errors="coerce")
        existing_keys = set(existing_ids.dropna().astype(int).tolist())
        return incremental[~incremental_ids.astype("Int64").isin(existing_keys)].copy()

    composite_cols = ["gameDate", "hometeamId", "awayteamId"]
    existing_keys = set(
        tuple(row) for row in existing[composite_cols].dropna().itertuples(index=False)
    )
    incremental_keys = incremental[composite_cols].dropna().itertuples(index=False)
    mask = [tuple(row) not in existing_keys for row in incremental_keys]
    return incremental.loc[mask].copy()


def _validate_required_fields(df: pd.DataFrame) -> pd.DataFrame:
    for column in REQUIRED_GAME_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    return df.dropna(subset=REQUIRED_GAME_COLUMNS).copy()


def run_incremental_pipeline(
    config_path: Optional[Path] = None,
    config: Optional[IncrementalIngestionConfig] = None,
) -> Optional[pd.DataFrame]:
    """
    Ingest new incremental files, append to raw games, and refresh features.
    """
    setup_logging(level="INFO")

    if config is None:
        if config_path is None:
            config_path = PROJECT_ROOT / "configs" / "incremental_ingestion.yaml"
        config = load_incremental_config(config_path)

    raw_incremental_dir = PROJECT_ROOT / config.raw_incremental_dir
    archive_dir = PROJECT_ROOT / config.archive_dir
    raw_games_path = PROJECT_ROOT / config.raw_games_path
    processed_games_path = PROJECT_ROOT / config.processed_games_path

    files = discover_incremental_files(raw_incremental_dir, config.max_files)
    if not files:
        logger.info("No incremental files found in %s", raw_incremental_dir)
        return None

    llm_client = None
    if config.use_llm and config.llm_provider.lower() == "openai":
        llm_client = OpenAIChatClient(
            model=config.llm_model, timeout_seconds=config.llm_timeout_seconds
        )

    mapper = LLMColumnMappingAgent(llm_client=llm_client, use_llm=config.use_llm)
    normalizer = RowNormalizationAgent()
    qa_agent = QualityAssuranceAgent()

    incremental_frames = []
    for path in files:
        logger.info("Loading incremental file: %s", path)
        frame = read_incremental_file(path)
        mapping = mapper.map_columns(frame)
        frame = frame.rename(columns=mapping)
        frame = normalizer.normalize(frame)
        incremental_frames.append(frame)

    incremental = pd.concat(incremental_frames, ignore_index=True)
    incremental = _validate_required_fields(incremental)
    qa_agent.validate(incremental)

    existing = _load_existing_games(raw_games_path)
    new_games = _dedupe_incremental(incremental, existing, config.dedupe_key)
    if new_games.empty:
        logger.info("No new games to append after deduplication.")
        archive_files(files, archive_dir)
        return None

    combined = pd.concat([existing, new_games], ignore_index=True)
    combined = combined.sort_values("gameDate")
    combined.to_csv(raw_games_path, index=False)
    logger.info("Appended %s new games to %s", len(new_games), raw_games_path)

    archive_files(files, archive_dir)

    parsed_games = parse_raw_games(combined)
    parsed_games["season"] = parsed_games["gameDate"].apply(get_nba_season)
    regular_season_games = filter_regular_season_games(parsed_games)
    regular_season_games.to_csv(processed_games_path, index=False)
    logger.info("Saved regular season games to %s", processed_games_path)

    if not config.update_features:
        return regular_season_games

    feature_config = load_experiment_config(PROJECT_ROOT / config.feature_config_path)
    lags = feature_config.feature_engineering.lags
    location_lags = feature_config.feature_engineering.location_lags

    teams_history = create_teams_history_table(
        input_file=str(PROJECT_ROOT / config.teams_history_path),
        output_file=None,
        current_season_year=config.current_season_year,
    )
    games_with_conference = add_conference(regular_season_games, teams_history)
    create_features_tables(
        games_with_conference,
        lags=lags,
        location_lags=location_lags,
    )
    final_features = merge_features(games_with_conference)
    final_features.to_csv(LOCAL_GAMES_FEATURES_PATH, index=False)
    logger.info("Saved merged features to %s", LOCAL_GAMES_FEATURES_PATH)

    return final_features


if __name__ == "__main__":
    run_incremental_pipeline()
