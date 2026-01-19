"""Data splitting utilities for train/validation/test sets."""

import logging
from typing import Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: Optional[int] = None,
    stratify: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets.

    Parameters
    ----------
    X : pd.DataFrame
        Features DataFrame
    y : pd.Series
        Target Series
    test_size : float, default=0.2
        Proportion of data to use for test set
    val_size : float, default=0.2
        Proportion of data to use for validation set (relative to remaining data after test split)
    random_state : int, optional
        Random seed for reproducibility
    stratify : pd.Series, optional
        Series to stratify on (e.g., target for classification)

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)

    Raises
    ------
    ValueError
        If test_size or val_size are invalid
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    if not 0 < val_size < 1:
        raise ValueError(f"val_size must be between 0 and 1, got {val_size}")

    if len(X) != len(y):
        raise ValueError(f"X and y must have the same length: {len(X)} vs {len(y)}")

    # First split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Adjust val_size to be relative to train_val set
    val_size_adjusted = val_size / (1 - test_size)

    # Second split: train vs val
    stratify_val = None
    if stratify is not None:
        stratify_val = stratify.loc[X_train_val.index]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=stratify_val,
    )

    logger.info(
        f"Split data: train={len(X_train)} ({len(X_train) / len(X) * 100:.1f}%), "
        f"val={len(X_val)} ({len(X_val) / len(X) * 100:.1f}%), "
        f"test={len(X_test)} ({len(X_test) / len(X) * 100:.1f}%)"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    date_column: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data temporally based on a date column.

    This ensures that training data comes before validation data,
    and validation data comes before test data.

    Parameters
    ----------
    X : pd.DataFrame
        Features DataFrame (must include date_column)
    y : pd.Series
        Target Series
    date_column : str
        Name of the date column to use for temporal splitting
    test_size : float, default=0.2
        Proportion of data to use for test set (most recent)
    val_size : float, default=0.2
        Proportion of data to use for validation set (middle period)

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)

    Raises
    ------
    ValueError
        If date_column is not found or dates are not sorted
    """
    if date_column not in X.columns:
        raise ValueError(f"Date column '{date_column}' not found in DataFrame")

    if len(X) != len(y):
        raise ValueError(f"X and y must have the same length: {len(X)} vs {len(y)}")

    # Sort by date
    dates = pd.to_datetime(X[date_column])
    sorted_indices = dates.sort_values().index

    X_sorted = X.loc[sorted_indices].copy()
    y_sorted = y.loc[sorted_indices].copy()

    # Calculate split indices
    n_total = len(X_sorted)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)
    n_train = n_total - n_test - n_val

    if n_train <= 0:
        raise ValueError(
            f"Invalid split sizes: train={n_train}, val={n_val}, test={n_test}. "
            f"Total samples: {n_total}"
        )

    # Split indices
    train_end = n_train
    val_end = n_train + n_val

    X_train = X_sorted.iloc[:train_end].copy()
    X_val = X_sorted.iloc[train_end:val_end].copy()
    X_test = X_sorted.iloc[val_end:].copy()

    y_train = y_sorted.iloc[:train_end].copy()
    y_val = y_sorted.iloc[train_end:val_end].copy()
    y_test = y_sorted.iloc[val_end:].copy()

    logger.info(
        f"Temporal split: train={len(X_train)} ({dates.iloc[:train_end].min()} to {dates.iloc[:train_end].max()}), "
        f"val={len(X_val)} ({dates.iloc[train_end:val_end].min()} to {dates.iloc[train_end:val_end].max()}), "
        f"test={len(X_test)} ({dates.iloc[val_end:].min()} to {dates.iloc[val_end:].max()})"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data with stratification to maintain class distribution.

    This is a convenience wrapper around train_val_test_split with stratify=y.

    Parameters
    ----------
    X : pd.DataFrame
        Features DataFrame
    y : pd.Series
        Target Series (used for stratification)
    test_size : float, default=0.2
        Proportion of data to use for test set
    val_size : float, default=0.2
        Proportion of data to use for validation set
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    return train_val_test_split(
        X=X,
        y=y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        stratify=y,
    )
