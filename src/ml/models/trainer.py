"""Model training utilities."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Literal
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


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

        Returns
        -------
        dict
            Training metrics
        """
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Evaluate on training set
        train_pred = self.model.predict(X_train)
        train_metrics = self._compute_metrics(y_train, train_pred, prefix="train")

        results = {"train": train_metrics}

        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_metrics = self._compute_metrics(y_val, val_pred, prefix="val")
            results["val"] = val_metrics

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

        cv_results = cross_validate(
            self.model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=return_train_score,
            n_jobs=-1,
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
            Scoring metric

        Returns
        -------
        BaseEstimator
            Best model from tuning
        """
        if scoring is None:
            if self.task_type == "regression":
                scoring = "neg_mean_squared_error"
            else:
                scoring = "accuracy"

        if method == "grid":
            search = GridSearchCV(
                self.model,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                random_state=self.random_state,
            )
        else:
            search = RandomizedSearchCV(
                self.model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                random_state=self.random_state,
            )

        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        self.search_results_ = search  # Store search object for accessing best_params
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
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")

        y_pred = self.model.predict(X)
        return self._compute_metrics(y, y_pred, prefix=prefix)

    def _compute_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Compute metrics based on task type."""
        metrics = {}

        if self.task_type == "regression":
            metrics[f"{prefix}_mse"] = mean_squared_error(y_true, y_pred)
            metrics[f"{prefix}_rmse"] = np.sqrt(metrics[f"{prefix}_mse"])
            metrics[f"{prefix}_mae"] = mean_absolute_error(y_true, y_pred)
            metrics[f"{prefix}_r2"] = r2_score(y_true, y_pred)
        else:
            metrics[f"{prefix}_accuracy"] = accuracy_score(y_true, y_pred)
            metrics[f"{prefix}_precision"] = precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
            metrics[f"{prefix}_recall"] = recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
            metrics[f"{prefix}_f1"] = f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            )

            # ROC AUC for binary classification
            if len(np.unique(y_true)) == 2:
                try:
                    # Note: X is not available here, need to pass it differently
                    # This is a limitation - ROC AUC calculation needs X
                    pass
                except (AttributeError, IndexError):
                    pass

        return metrics


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
