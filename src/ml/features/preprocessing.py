"""Preprocessing transformers for feature engineering."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    OneHotEncoder,
    FunctionTransformer,
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from typing import Optional, List, Union, Literal


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select specific features from a DataFrame."""

    def __init__(self, columns: List[str]):
        """
        Initialize FeatureSelector.

        Parameters
        ----------
        columns : list of str
            List of column names to select
        """
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer (no-op for feature selection)."""
        missing_cols = set(self.columns) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Columns not found in X: {missing_cols}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select specified columns."""
        return X[self.columns].copy()

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return self.columns


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """Handle missing values in numerical and categorical columns."""

    def __init__(
        self,
        strategy: Union[
            Literal["mean", "median", "most_frequent", "constant"], str
        ] = "mean",
        fill_value: Optional[Union[str, int, float]] = None,
        columns: Optional[List[str]] = None,
    ):
        """
        Initialize MissingValueImputer.

        Parameters
        ----------
        strategy : str, default='mean'
            Imputation strategy ('mean', 'median', 'most_frequent', 'constant')
        fill_value : str, int, or float, optional
            Value to use when strategy='constant'
        columns : list of str, optional
            Specific columns to impute. If None, imputes all columns.
        """
        self.strategy = strategy
        self.fill_value = fill_value
        self.columns = columns
        self.imputers_ = {}
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y=None):
        """Fit imputers on training data."""
        if self.columns is None:
            self.columns = X.columns.tolist()

        self.feature_names_ = X.columns.tolist()

        for col in self.columns:
            if col not in X.columns:
                continue

            if X[col].dtype in ["object", "category"]:
                # For categorical, use most_frequent or constant
                strategy = (
                    "most_frequent" if self.strategy != "constant" else "constant"
                )
                self.imputers_[col] = SimpleImputer(
                    strategy=strategy,
                    fill_value=self.fill_value if self.strategy == "constant" else None,
                )
            else:
                # For numerical
                self.imputers_[col] = SimpleImputer(
                    strategy=self.strategy,
                    fill_value=self.fill_value if self.strategy == "constant" else None,
                )

            self.imputers_[col].fit(X[[col]])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by imputing missing values."""
        X_transformed = X.copy()

        for col in self.columns:
            if col in X.columns and col in self.imputers_:
                X_transformed[col] = self.imputers_[col].transform(X[[col]]).ravel()

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return (
            self.feature_names_ if self.feature_names_ is not None else input_features
        )


class OutlierHandler(BaseEstimator, TransformerMixin):
    """Handle outliers using IQR method or clipping."""

    def __init__(
        self,
        method: Literal["clip", "remove"] = "clip",
        columns: Optional[List[str]] = None,
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99,
    ):
        """
        Initialize OutlierHandler.

        Parameters
        ----------
        method : {'clip', 'remove'}, default='clip'
            Method to handle outliers
        columns : list of str, optional
            Specific columns to process. If None, processes all numerical columns.
        lower_percentile : float, default=0.01
            Lower percentile for outlier detection
        upper_percentile : float, default=0.99
            Upper percentile for outlier detection
        """
        self.method = method
        self.columns = columns
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y=None):
        """Fit outlier handler on training data."""
        if self.columns is None:
            # Select only numerical columns
            self.columns = X.select_dtypes(include=[np.number]).columns.tolist()

        self.feature_names_ = X.columns.tolist()

        for col in self.columns:
            if col not in X.columns:
                continue

            if X[col].dtype in [np.number]:
                self.lower_bounds_[col] = X[col].quantile(self.lower_percentile)
                self.upper_bounds_[col] = X[col].quantile(self.upper_percentile)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by handling outliers."""
        X_transformed = X.copy()

        for col in self.columns:
            if col not in X.columns or col not in self.lower_bounds_:
                continue

            if self.method == "clip":
                X_transformed[col] = X_transformed[col].clip(
                    lower=self.lower_bounds_[col],
                    upper=self.upper_bounds_[col],
                )
            elif self.method == "remove":
                mask = (X_transformed[col] >= self.lower_bounds_[col]) & (
                    X_transformed[col] <= self.upper_bounds_[col]
                )
                X_transformed = X_transformed[mask]

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        return (
            self.feature_names_ if self.feature_names_ is not None else input_features
        )


def create_preprocessing_pipeline(
    numerical_features: List[str],
    categorical_features: Optional[List[str]] = None,
    boolean_features: Optional[List[str]] = None,
    scaling_method: Literal["standard", "robust", "minmax"] = "standard",
    imputation_strategy: str = "mean",
    handle_outliers: bool = True,
) -> ColumnTransformer:
    """
    Create a preprocessing pipeline for numerical, categorical, and boolean features.

    Parameters
    ----------
    numerical_features : list of str
        List of numerical feature column names
    categorical_features : list of str, optional
        List of categorical feature column names
    boolean_features : list of str, optional
        List of boolean feature column names. Boolean features are handled separately
        from numerical features because SimpleImputer with mean/median strategy
        doesn't support boolean dtype.
    scaling_method : {'standard', 'robust', 'minmax'}, default='standard'
        Scaling method for numerical features
    imputation_strategy : str, default='mean'
        Imputation strategy ('mean', 'median', 'most_frequent')
    handle_outliers : bool, default=True
        Whether to handle outliers

    Returns
    -------
    ColumnTransformer
        Preprocessing pipeline
    """
    transformers = []
    from sklearn.pipeline import Pipeline

    if boolean_features is None:
        boolean_features = []

    # Numerical features pipeline
    if numerical_features:
        num_steps = []

        if handle_outliers:
            num_steps.append(
                ("outlier_handler", OutlierHandler(columns=numerical_features))
            )

        num_steps.append(("imputer", SimpleImputer(strategy=imputation_strategy)))

        if scaling_method == "standard":
            num_steps.append(("scaler", StandardScaler()))
        elif scaling_method == "robust":
            num_steps.append(("scaler", RobustScaler()))
        elif scaling_method == "minmax":
            num_steps.append(("scaler", MinMaxScaler()))

        transformers.append(("numerical", Pipeline(num_steps), numerical_features))

    # Boolean features pipeline (treat as binary categorical)
    if boolean_features:
        bool_steps = [
            # Convert bool to int, then impute with most_frequent
            (
                "bool_to_int",
                FunctionTransformer(
                    func=lambda x: x.astype(int),
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
        transformers.append(("boolean", Pipeline(bool_steps), boolean_features))

    # Categorical features pipeline
    if categorical_features:
        cat_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    drop="if_binary",  # Drop one column for binary features to avoid multicollinearity
                    sparse_output=False,  # Return dense arrays for compatibility
                    handle_unknown="ignore",  # Ignore unknown categories in test set
                ),
            ),
        ]
        transformers.append(("categorical", Pipeline(cat_steps), categorical_features))

    ct = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
    )
    ct.set_output(transform="pandas")
    return ct
