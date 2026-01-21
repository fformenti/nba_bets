"""Data loading utilities with validation and error handling."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def load_dataframe(
    file_path: Path,
    parse_dates: Optional[list] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Load a DataFrame from a CSV file with validation.

    Parameters
    ----------
    file_path : Path
        Path to the CSV file
    parse_dates : list, optional
        List of column names to parse as dates
    **kwargs
        Additional arguments passed to pd.read_csv

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    ValueError
        If the file is empty or cannot be parsed
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    try:
        df = pd.read_csv(file_path, parse_dates=parse_dates, **kwargs)
    except Exception as e:
        raise ValueError(f"Error reading CSV file {file_path}: {e}") from e

    if df.empty:
        raise ValueError(f"Data file is empty: {file_path}")

    logger.info(
        f"Loaded DataFrame from {file_path}: {len(df)} rows, {len(df.columns)} columns"
    )
    return df


def load_features(
    file_path: Path,
    target_column: str,
    feature_columns: Optional[list] = None,
    drop_na: bool = True,
    parse_dates: Optional[list] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load features and target from a CSV file.

    Parameters
    ----------
    file_path : Path
        Path to the CSV file
    target_column : str
        Name of the target column
    feature_columns : list, optional
        List of feature column names. If None, uses all columns except target.
    drop_na : bool, default=True
        Whether to drop rows with missing values
    parse_dates : list, optional
        List of column names to parse as dates

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Features DataFrame and target Series

    Raises
    ------
    ValueError
        If target_column is not found in the DataFrame
    """
    df = load_dataframe(file_path, parse_dates=parse_dates)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    # Extract target
    y = df[target_column].copy()

    # Extract features
    if feature_columns is None:
        X = df.drop(columns=[target_column])
    else:
        missing_cols = set(feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Feature columns not found in DataFrame: {missing_cols}")
        X = df[feature_columns].copy()

    # Drop missing values if requested
    if drop_na:
        initial_len = len(X)
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask].copy()
        y = y[mask].copy()
        dropped = initial_len - len(X)
        if dropped > 0:
            logger.info(
                f"Dropped {dropped} rows with missing values ({dropped / initial_len * 100:.1f}%)"
            )

    logger.info(
        f"Loaded features: {len(X)} samples, {len(X.columns)} features. "
        f"Target distribution: {y.value_counts().to_dict()}"
    )

    return X, y


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[list] = None,
    min_rows: int = 1,
    check_dtypes: Optional[dict] = None,
) -> None:
    """
    Validate DataFrame structure and content.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : list, optional
        List of required column names
    min_rows : int, default=1
        Minimum number of rows required
    check_dtypes : dict, optional
        Dictionary mapping column names to expected dtypes

    Raises
    ------
    ValueError
        If validation fails
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    if len(df) < min_rows:
        raise ValueError(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")

    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    if check_dtypes:
        for col, expected_dtype in check_dtypes.items():
            if col not in df.columns:
                continue
            actual_dtype = df[col].dtype
            if not np.issubdtype(actual_dtype, expected_dtype):
                raise ValueError(
                    f"Column '{col}' has dtype {actual_dtype}, expected {expected_dtype}"
                )
