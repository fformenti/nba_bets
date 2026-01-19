"""Data validation utilities."""

import logging
from typing import Optional, List, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def validate_data_shapes(
    X: pd.DataFrame,
    y: pd.Series,
    min_samples: int = 1,
) -> None:
    """
    Validate that X and y have compatible shapes.

    Parameters
    ----------
    X : pd.DataFrame
        Features DataFrame
    y : pd.Series
        Target Series
    min_samples : int, default=1
        Minimum number of samples required

    Raises
    ------
    ValueError
        If shapes are incompatible or insufficient samples
    """
    if len(X) != len(y):
        raise ValueError(f"X and y must have the same length: {len(X)} vs {len(y)}")

    if len(X) < min_samples:
        raise ValueError(f"Insufficient samples: {len(X)} < {min_samples}")


def validate_target_distribution(
    y: pd.Series,
    task_type: str,
    min_samples_per_class: int = 1,
) -> None:
    """
    Validate target distribution for classification tasks.

    Parameters
    ----------
    y : pd.Series
        Target Series
    task_type : str
        Task type ('classification' or 'regression')
    min_samples_per_class : int, default=1
        Minimum samples per class for classification

    Raises
    ------
    ValueError
        If target distribution is invalid
    """
    if task_type == "classification":
        value_counts = y.value_counts()
        if len(value_counts) < 2:
            raise ValueError(
                f"Classification requires at least 2 classes, found {len(value_counts)}"
            )

        min_count = value_counts.min()
        if min_count < min_samples_per_class:
            raise ValueError(
                f"Class with {min_count} samples is below minimum {min_samples_per_class}"
            )

        logger.info(f"Target distribution: {value_counts.to_dict()}")


def validate_feature_types(
    X: pd.DataFrame,
    expected_numerical: Optional[List[str]] = None,
    expected_categorical: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Validate and identify feature types.

    Parameters
    ----------
    X : pd.DataFrame
        Features DataFrame
    expected_numerical : list, optional
        Expected numerical feature names
    expected_categorical : list, optional
        Expected categorical feature names

    Returns
    -------
    tuple
        (numerical_features, categorical_features)

    Raises
    ------
    ValueError
        If expected features are missing or have wrong types
    """
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if expected_numerical:
        missing = set(expected_numerical) - set(numerical_features)
        if missing:
            raise ValueError(f"Missing expected numerical features: {missing}")

    if expected_categorical:
        missing = set(expected_categorical) - set(categorical_features)
        if missing:
            raise ValueError(f"Missing expected categorical features: {missing}")

    return numerical_features, categorical_features


def check_for_missing_values(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Check for missing values and report statistics.

    Parameters
    ----------
    X : pd.DataFrame
        Features DataFrame
    y : pd.Series, optional
        Target Series
    threshold : float, default=1.0
        Maximum fraction of missing values allowed per column

    Returns
    -------
    pd.DataFrame
        Summary of missing values

    Raises
    ------
    ValueError
        If missing value fraction exceeds threshold
    """
    missing_stats = []

    for col in X.columns:
        missing_count = X[col].isna().sum()
        missing_pct = missing_count / len(X) * 100
        missing_stats.append(
            {
                "column": col,
                "missing_count": missing_count,
                "missing_percentage": missing_pct,
            }
        )

        if missing_pct > threshold * 100:
            logger.warning(
                f"Column '{col}' has {missing_pct:.1f}% missing values "
                f"(threshold: {threshold * 100:.1f}%)"
            )

    if y is not None:
        missing_count = y.isna().sum()
        missing_pct = missing_count / len(y) * 100
        missing_stats.append(
            {
                "column": "target",
                "missing_count": missing_count,
                "missing_percentage": missing_pct,
            }
        )

        if missing_pct > threshold * 100:
            raise ValueError(
                f"Target has {missing_pct:.1f}% missing values "
                f"(threshold: {threshold * 100:.1f}%)"
            )

    missing_df = pd.DataFrame(missing_stats)
    total_missing = missing_df["missing_count"].sum()

    if total_missing > 0:
        logger.info(
            f"Missing values summary:\n{missing_df[missing_df['missing_count'] > 0]}"
        )

    return missing_df


def validate_split_sizes(
    train_size: int,
    val_size: int,
    test_size: int,
    total_size: int,
) -> None:
    """
    Validate that split sizes are reasonable.

    Parameters
    ----------
    train_size : int
        Training set size
    val_size : int
        Validation set size
    test_size : int
        Test set size
    total_size : int
        Total dataset size

    Raises
    ------
    ValueError
        If split sizes are invalid
    """
    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        raise ValueError(
            f"All split sizes must be > 0: train={train_size}, val={val_size}, test={test_size}"
        )

    if train_size + val_size + test_size != total_size:
        raise ValueError(
            f"Split sizes don't sum to total: {train_size} + {val_size} + {test_size} != {total_size}"
        )

    train_pct = train_size / total_size * 100
    val_pct = val_size / total_size * 100
    test_pct = test_size / total_size * 100

    if train_pct < 50:
        logger.warning(
            f"Training set is only {train_pct:.1f}% of data (recommended: >=60%)"
        )

    if test_pct < 10:
        logger.warning(f"Test set is only {test_pct:.1f}% of data (recommended: >=15%)")
