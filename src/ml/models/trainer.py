"""Model training utilities."""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Literal

from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    TimeSeriesSplit,
    cross_validate,
)

from src.ml.evaluation.metrics import (
    compute_regression_metrics,
    compute_classification_metrics,
    format_metrics_line,
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Utility class for training and evaluating ML models."""

    def __init__(
        self,
        model: BaseEstimator,
        task_type: Literal["regression", "classification"],
        random_state: Optional[int] = None,
    ):
        """
        Initialize ModelTrainer.

        Parameters
        ----------
        model : BaseEstimator
            Scikit-learn model or pipeline
        task_type : {'regression', 'classification'}
            Type of ML task
        random_state : int, optional
            Random seed for reproducibility
        """
        self.model = model
        self.task_type = task_type
        self.random_state = random_state
        self.is_fitted = False

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_val : pd.DataFrame, optional
            Validation features
        y_val : pd.Series, optional
            Validation target
        sample_weight : np.ndarray, optional
            Per-sample weights for training

        Returns
        -------
        dict
            Training metrics with keys 'train' and optionally 'val'

        Raises
        ------
        ValueError
            If X_val is provided without y_val or vice versa
        """
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must both be provided or both be None")

        logger.info(f"Training {self.task_type} model on {len(X_train)} samples")
        fit_params: Dict[str, Any] = {}
        if sample_weight is not None:
            fit_params["model__sample_weight"] = sample_weight
        self.model.fit(X_train, y_train, **fit_params)
        self.is_fitted = True

        # Evaluate on training set
        train_pred = self.model.predict(X_train)
        train_metrics = self._compute_metrics(
            y_train, train_pred, X_train, prefix="train"
        )

        results = {"train": train_metrics}

        train_line = format_metrics_line(train_metrics, prefix="train")
        if train_line:
            logger.info(f"Train:      {train_line}")

        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_metrics = self._compute_metrics(y_val, val_pred, X_val, prefix="val")
            results["val"] = val_metrics
            val_line = format_metrics_line(val_metrics, prefix="val")
            if val_line:
                logger.info(f"Validation: {val_line}")

        return results

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: Optional[str] = None,
        return_train_score: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.

        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        cv : int, default=5
            Number of folds
        scoring : str, optional
            Scoring metric. If None, uses default for task type.
        return_train_score : bool, default=True
            Whether to return training scores

        Returns
        -------
        dict
            Cross-validation results
        """
        if scoring is None:
            if self.task_type == "regression":
                scoring = "neg_mean_squared_error"
            else:
                scoring = "accuracy"

        logger.info(f"Performing {cv}-fold cross-validation with scoring={scoring}")

        cv_results = cross_validate(
            self.model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=return_train_score,
            n_jobs=-1,
        )

        logger.info(
            f"CV results - Test score: {cv_results['test_score'].mean():.4f} "
            f"(±{cv_results['test_score'].std():.4f})"
        )

        return cv_results

    def hyperparameter_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Dict[str, Any],
        method: Literal["grid", "random"] = "grid",
        cv: int = 5,
        n_iter: int = 100,
        scoring: Optional[str] = None,
        cv_strategy: str = "timeseries",
        sample_weight: Optional[np.ndarray] = None,
    ) -> BaseEstimator:
        """
        Perform hyperparameter tuning.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        param_grid : dict
            Parameter grid for tuning
        method : {'grid', 'random'}, default='grid'
            Search method
        cv : int, default=5
            Number of CV folds
        n_iter : int, default=100
            Number of iterations for random search
        scoring : str, optional
            Scoring metric. If None, uses default for task type.
        cv_strategy : str, default='timeseries'
            CV strategy: 'timeseries' (TimeSeriesSplit) or 'kfold' (standard k-fold).
        sample_weight : np.ndarray, optional
            Per-sample weights passed to fit via model__sample_weight.

        Returns
        -------
        BaseEstimator
            Best model from tuning

        Raises
        ------
        ValueError
            If method is not 'grid' or 'random'
        """
        if scoring is None:
            if self.task_type == "regression":
                scoring = "neg_mean_squared_error"
            else:
                scoring = "accuracy"

        if method not in ["grid", "random"]:
            raise ValueError(f"method must be 'grid' or 'random', got {method}")

        cv_splitter = TimeSeriesSplit(n_splits=cv) if cv_strategy == "timeseries" else cv

        logger.info(
            f"Starting {method} search hyperparameter tuning: "
            f"cv={cv} ({cv_strategy}), scoring={scoring}, n_iter={n_iter if method == 'random' else 'N/A'}"
        )

        if method == "grid":
            search = GridSearchCV(
                self.model,
                param_grid,
                cv=cv_splitter,
                scoring=scoring,
                n_jobs=-1,
                refit=False,
                return_train_score=True,
            )
        else:
            search = RandomizedSearchCV(
                self.model,
                param_grid,
                n_iter=n_iter,
                cv=cv_splitter,
                scoring=scoring,
                n_jobs=-1,
                random_state=self.random_state,
                refit=False,
                return_train_score=True,
            )

        # HP selection: don't pass sample_weight so CV scores reflect
        # unweighted performance (the weights affect training, not evaluation).
        search.fit(X_train, y_train)
        self.search_results_ = search

        logger.info(
            f"Hyperparameter tuning completed. Best score: {search.best_score_:.4f}, "
            f"Best params: {search.best_params_}"
        )

        # Manually refit with best params and sample weights
        self.model.set_params(**search.best_params_)
        fit_params: Dict[str, Any] = {}
        if sample_weight is not None:
            fit_params["model__sample_weight"] = sample_weight
        self.model.fit(X_train, y_train, **fit_params)
        self.is_fitted = True

        return self.model

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        prefix: str = "test",
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series
            True target values
        prefix : str, default='test'
            Prefix for metric names

        Returns
        -------
        dict
            Evaluation metrics

        Raises
        ------
        ValueError
            If model has not been trained
        """
        if not self.is_fitted:
            raise ValueError(
                "Model must be trained before evaluation. Call train() first."
            )

        logger.debug(f"Evaluating model on {len(X)} samples (prefix: {prefix})")
        y_pred = self.model.predict(X)
        metrics = self._compute_metrics(y, y_pred, X, prefix=prefix)
        line = format_metrics_line(metrics, prefix=prefix)
        if line:
            logger.info(f"{prefix.capitalize()}: {line}")
        return metrics

    def _compute_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        X: pd.DataFrame,
        prefix: str = "",
    ) -> Dict[str, float]:
        """
        Compute metrics based on task type.

        Parameters
        ----------
        y_true : pd.Series
            True target values
        y_pred : np.ndarray
            Predicted values
        X : pd.DataFrame
            Features (needed for probability predictions in classification)
        prefix : str, default=''
            Prefix for metric names

        Returns
        -------
        dict
            Dictionary of metrics with prefix
        """
        if self.task_type == "regression":
            base_metrics = compute_regression_metrics(y_true, y_pred)
        else:
            # Get predicted probabilities for classification metrics
            y_pred_proba = None
            if hasattr(self.model, "predict_proba"):
                try:
                    y_pred_proba = self.model.predict_proba(X)
                except Exception as e:
                    logger.warning(f"Could not compute probabilities: {e}")

            base_metrics = compute_classification_metrics(
                y_true, y_pred, y_pred_proba=y_pred_proba
            )

        # Add prefix to metric names
        return {f"{prefix}_{k}": v for k, v in base_metrics.items()}


def train_model(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task_type: Literal["regression", "classification"],
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    random_state: Optional[int] = None,
) -> tuple[BaseEstimator, Dict[str, Any]]:
    """
    Convenience function to train a model.

    Parameters
    ----------
    model : BaseEstimator
        Scikit-learn model
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    task_type : {'regression', 'classification'}
        Type of ML task
    X_val : pd.DataFrame, optional
        Validation features
    y_val : pd.Series, optional
        Validation target
    random_state : int, optional
        Random seed

    Returns
    -------
    tuple
        Trained model and training metrics
    """
    trainer = ModelTrainer(model, task_type, random_state)
    metrics = trainer.train(X_train, y_train, X_val, y_val)
    return trainer.model, metrics


def evaluate_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: Literal["regression", "classification"],
    prefix: str = "test",
) -> Dict[str, float]:
    """
    Convenience function to evaluate a model.

    Parameters
    ----------
    model : BaseEstimator
        Trained scikit-learn model
    X : pd.DataFrame
        Features
    y : pd.Series
        True target values
    task_type : {'regression', 'classification'}
        Type of ML task
    prefix : str, default='test'
        Prefix for metric names

    Returns
    -------
    dict
        Evaluation metrics
    """
    trainer = ModelTrainer(model, task_type)
    trainer.is_fitted = True
    trainer.model = model
    return trainer.evaluate(X, y, prefix)
