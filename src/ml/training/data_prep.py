from pathlib import Path
from typing import Optional

import pandas as pd

from src.ml.datasets.loaders import load_dataframe, validate_dataframe
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def filter_minimum_games_played(df: pd.DataFrame, minimum_games: int = 15) -> pd.DataFrame:
    return df[
        (df["games_played_HT"] > minimum_games)
        & (df["games_played_VT"] > minimum_games)
    ]


def load_and_validate_data(
    data_path: Path,
    target_column: str,
    date_column: Optional[str] = None,
) -> pd.DataFrame:
    logger.info(f"Loading data from {data_path}")
    df = load_dataframe(data_path, parse_dates=[date_column] if date_column else None)
    validate_dataframe(
        df,
        required_columns=[target_column] + ([date_column] if date_column else []),
        min_rows=100,
    )
    return df


def prepare_data(
    df: pd.DataFrame,
    target_column: str,
    drop_na: bool = True,
    metadata_columns: Optional[list] = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if drop_na:
        initial_len = len(df)
        df = df.dropna().copy()
        dropped = initial_len - len(df)
        if dropped > 0:
            logger.info(
                f"Dropped {dropped} rows with missing values ({dropped / initial_len * 100:.1f}%)"
            )

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    y = df[target_column].copy()

    if metadata_columns:
        metadata_cols = [col for col in metadata_columns if col in df.columns]
        metadata = df[metadata_cols].copy()
    else:
        metadata = pd.DataFrame(index=df.index)

    logger.info(
        f"Loaded {len(df)} samples. Target distribution:\n{y.value_counts().to_dict()}"
    )

    return df, y, metadata
