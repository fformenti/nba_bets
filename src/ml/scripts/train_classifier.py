"""
Refactored training script for classification models.

This script demonstrates best practices:
- Configuration management via YAML
- Proper logging instead of print statements
- Modular feature engineering
- Input validation and error handling
- Clean separation of concerns
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.logging_config import setup_logging, get_logger
from src.ml.datasets.loaders import load_dataframe, validate_dataframe
from src.ml.datasets.splitters import (
    temporal_split,
    train_val_test_split,
    stratified_split,
)
from src.ml.features.engineering import (
    create_conference_delta,
    get_home_conference_vs_away_conference_record,
    create_delta_features,
    identify_feature_types,
)
from src.ml.features.preprocessing import create_preprocessing_pipeline
from src.ml.models.trainer import ModelTrainer
from src.ml.models.registry import ModelRegistry
from src.ml.models.baseline import PointDifferentialBaseline, RecordDifferenceBaseline
from src.ml.evaluation.metrics import print_metrics_summary
from src.ml.evaluation.visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
)
from src.ml.config.loader import load_experiment_config
from src.ml.config.schema import ExperimentConfig
from src.ml.tracking import MLflowTracker
from src.config import PROJECT_ROOT, LOCAL_GAMES_FEATURES_PATH

logger = get_logger(__name__)


def clean_feature_names(feature_names: list) -> list:
    """
    Remove ColumnTransformer prefixes (e.g., 'numerical__', 'categorical__') from feature names.

    ColumnTransformer prefixes feature names with the transformer name followed by '__'.
    This function removes those prefixes to get clean feature names.

    Parameters
    ----------
    feature_names : list
        List of feature names that may contain prefixes

    Returns
    -------
    list
        Feature names with prefixes removed
    """
    cleaned = []
    for name in feature_names:
        # Remove ColumnTransformer prefixes (format: "transformer_name__feature_name")
        # Split on '__' and take everything after the first occurrence
        if "__" in name:
            # Split on '__' and join everything after the first part
            parts = name.split("__", 1)
            if len(parts) == 2:
                cleaned.append(parts[1])
            else:
                cleaned.append(name)
        else:
            cleaned.append(name)
    return cleaned


def filter_minimun_games_played(
    df: pd.DataFrame, minimum_games: int = 15
) -> pd.DataFrame:
    """
    Filter data based on minimum games played.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    minimum_games : int
        Minimum games played
    """
    return df[
        (df["games_played_HT"] > minimum_games)
        & (df["games_played_VT"] > minimum_games)
    ]


def load_and_validate_data(
    data_path: Path,
    target_column: str,
    date_column: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load and validate data.
    """
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
    """
    Load and prepare data for training.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    drop_na : bool, default=True
        Whether to drop rows with missing values
    metadata_columns : list, optional
        List of metadata columns to preserve

    Returns
    -------
    tuple
        (features_df, target_series, metadata_df)
    """

    # Drop missing values if requested
    if drop_na:
        initial_len = len(df)
        df = df.dropna().copy()
        dropped = initial_len - len(df)
        if dropped > 0:
            logger.info(
                f"Dropped {dropped} rows with missing values ({dropped / initial_len * 100:.1f}%)"
            )

    # Extract target
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    y = df[target_column].copy()

    # Extract metadata
    if metadata_columns:
        metadata_cols = [col for col in metadata_columns if col in df.columns]
        metadata = df[metadata_cols].copy()
    else:
        metadata = pd.DataFrame(index=df.index)

    logger.info(
        f"Loaded {len(df)} samples. Target distribution:\n{y.value_counts().to_dict()}"
    )

    return df, y, metadata


def create_model(
    model_name: str,
    model_config: Dict[str, Any],
    task_type: str,
    random_state: Optional[int] = None,
) -> tuple[BaseEstimator, Dict[str, Any]]:
    """
    Create a model instance based on configuration.

    Parameters
    ----------
    model_name : str
        Name of the model ('random_forest' or 'gradient_boosting')
    model_config : dict
        Model-specific configuration
    task_type : str
        Task type ('classification' or 'regression')
    random_state : int, optional
        Random seed

    Returns
    -------
    tuple
        (model_instance, param_grid_for_tuning)
    """
    if model_name == "random_forest":
        rf_config = model_config.get("random_forest", {})
        model = RandomForestClassifier(
            n_estimators=rf_config.get("n_estimators", 100),
            max_depth=rf_config.get("max_depth"),
            min_samples_split=rf_config.get("min_samples_split", 20),
            min_samples_leaf=rf_config.get("min_samples_leaf", 10),
            max_features=rf_config.get("max_features", "sqrt"),
            class_weight=rf_config.get("class_weight", "balanced"),
            random_state=random_state,
            n_jobs=-1,
        )
        param_grid = {
            "model__n_estimators": [50, 100, 200, 300],
            "model__max_depth": [5, 7, 10, 15, None],
            "model__min_samples_split": [10, 20, 30, 50],
            "model__min_samples_leaf": [5, 10, 15, 20],
            "model__max_features": ["sqrt", "log2", 0.5, 0.7],
            "model__class_weight": ["balanced", "balanced_subsample", None],
        }

    elif model_name == "gradient_boosting":
        gb_config = model_config.get("gradient_boosting", {})
        model = GradientBoostingClassifier(
            n_estimators=gb_config.get("n_estimators", 100),
            learning_rate=gb_config.get("learning_rate", 0.1),
            max_depth=gb_config.get("max_depth", 5),
            min_samples_split=gb_config.get("min_samples_split", 20),
            min_samples_leaf=gb_config.get("min_samples_leaf", 10),
            random_state=random_state,
        )
        param_grid = {
            "model__n_estimators": [50, 100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__max_depth": [3, 5, 7, 10],
            "model__min_samples_split": [10, 20, 30, 50],
            "model__min_samples_leaf": [5, 10, 15, 20],
            "model__subsample": [0.8, 0.9, 1.0],
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    logger.info(f"Created {model_name} model")
    return model, param_grid


def train_model_with_config(
    model_name: str,
    model_config: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    preprocessor: Any,
    random_state: Optional[int] = None,
) -> tuple[Pipeline, ModelTrainer, Dict[str, Any]]:
    """
    Train a model with optional hyperparameter tuning.

    Parameters
    ----------
    model_name : str
        Model name
    model_config : dict
        Model configuration
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation target
    preprocessor : Any
        Preprocessing pipeline
    random_state : int, optional
        Random seed

    Returns
    -------
    tuple
        (trained_pipeline, trainer, training_results)
    """
    # Create model
    model, param_grid = create_model(
        model_name, model_config, task_type="classification", random_state=random_state
    )

    # Create pipeline
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # Create trainer
    trainer = ModelTrainer(
        model=pipeline,
        task_type="classification",
        random_state=random_state,
    )

    # Hyperparameter tuning if enabled
    hp_config = model_config.get("hyperparameter_tuning", {})
    if hp_config.get("enabled", False):
        logger.info(f"Starting hyperparameter tuning for {model_name}")
        trainer.hyperparameter_tuning(
            X_train=X_train,
            y_train=y_train,
            param_grid=param_grid,
            method=hp_config.get("method", "random"),
            cv=hp_config.get("cv_folds", 5),
            n_iter=hp_config.get("n_iter", 100),
            scoring=hp_config.get("scoring", "f1"),
        )
        pipeline = trainer.model

    # Train model
    logger.info(f"Training {model_name}...")
    training_results = trainer.train(X_train, y_train, X_val, y_val)

    return pipeline, trainer, training_results


def train_baseline_model(
    baseline_model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    try:
        baseline_model.fit(X_train, y_train)

        # Compute metrics using trainer for consistency
        baseline_trainer = ModelTrainer(
            model=baseline_model,
            task_type="classification",
        )
        baseline_trainer.is_fitted = True

        baseline_train_metrics = baseline_trainer.evaluate(
            X_train, y_train, prefix="train"
        )
        baseline_val_metrics = baseline_trainer.evaluate(X_val, y_val, prefix="val")
        baseline_test_metrics = baseline_trainer.evaluate(X_test, y_test, prefix="test")
        logger.info(
            f"{baseline_model.name_caption} Test metrics: {baseline_test_metrics}"
        )

        return {
            "pipeline": baseline_model,
            "trainer": baseline_trainer,
            "training_results": {
                "train": baseline_train_metrics,
                "val": baseline_val_metrics,
            },
            "test_metrics": baseline_test_metrics,
        }

    except Exception as e:
        logger.error(
            f"Error training {baseline_model.name_caption}: {e}", exc_info=True
        )


def main(
    config_path: Optional[Path] = None, experiment_name: str = "nba_bets_classification"
):
    """
    Main training function.

    Parameters
    ----------
    config_path : Path, optional
        Path to configuration YAML file. If None, uses default configuration.
    experiment_name : str, default='nba_bets_classification'
        MLflow experiment name for tracking
    """
    # Setup logging
    setup_logging(level="INFO")

    # Load configuration
    if config_path is None:
        # config_path = PROJECT_ROOT / "configs" / "train_classification_example.yaml"
        config_path = PROJECT_ROOT / "configs" / "my_experiment.yaml"

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        config = ExperimentConfig()
    else:
        config = load_experiment_config(config_path)

    # Start MLflow tracking
    with MLflowTracker(
        experiment_name=experiment_name,
        run_name=config_path.stem if config_path else None,
    ) as tracker:
        # Log configuration
        tracker.log_config(config.model_dump())

        # Extract configuration with defaults
        data_config = config.data
        split_config = config.splitting
        model_config = config.model.model_dump()
        eval_config = config.evaluation
        paths_config = config.paths
        feat_eng_config = config.feature_engineering
        filters_config = config.filters

        # Data loading
        data_path = (
            Path(data_config.path)
            if data_config.path
            else Path(LOCAL_GAMES_FEATURES_PATH)
        )
        target_column = data_config.target_column
        date_column = data_config.date_column

        # feature engineering configuration
        lags = feat_eng_config.lags
        location_lags = feat_eng_config.location_lags
        metadata_columns = feat_eng_config.metadata_columns
        originally_enriched_columns = feat_eng_config.originally_enriched_columns

        # filters configuration
        minimum_games = filters_config.minimum_games

        # model configuration
        only_different_conferences = True
        only_same_conferences = False
        within_and_across_conferences = False

        # Log key parameters to MLflow
        tracker.log_params(
            {
                "target_column": target_column,
                "split_method": split_config.method,
                "test_size": split_config.test_size,
                "val_size": split_config.val_size,
                "minimum_games": minimum_games,
                "only_different_conferences": only_different_conferences,
                "model_name": model_config.get("name", "random_forest"),
                "scaling_method": config.preprocessing.scaling_method,
            }
        )

        # Only one can be True
        if (
            sum(
                [
                    only_different_conferences,
                    only_same_conferences,
                    within_and_across_conferences,
                ]
            )
            != 1
        ):
            raise ValueError(
                "Only one of only_different_conferences, only_same_conferences, or within_and_across_conferences can be True"
            )

        games_enriched = load_and_validate_data(
            data_path=data_path,
            target_column=target_column,
            date_column=date_column,
        )

        # Subset only games where teams are from different conferences
        if only_different_conferences:
            games_enriched = games_enriched[
                games_enriched["hometeamConference"]
                != games_enriched["awayteamConference"]
            ]
        if only_same_conferences:
            games_enriched = games_enriched[
                games_enriched["hometeamConference"]
                == games_enriched["awayteamConference"]
            ]

        # Minimum games played
        games_enriched = filter_minimun_games_played(games_enriched, minimum_games)

        df, y, metadata = prepare_data(
            df=games_enriched,
            target_column=target_column,
            drop_na=data_config.drop_na,
            metadata_columns=metadata_columns,
        )

        # Combine features
        df = create_delta_features(df, lags, location_lags)

        # Combine conference features (not necessary if only within conferences)
        if only_different_conferences:
            # This will create the features: home_conference_vs_away_conference_record and games_played_at_home_conference
            df = get_home_conference_vs_away_conference_record(df)
        if within_and_across_conferences:
            # This will create the feature: conference_diff_east_pct
            df = create_conference_delta(df)

        # Select features for model
        exclude_cols = feat_eng_config.exclude_columns

        # Identify feature types
        feature_types = identify_feature_types(df, exclude_columns=exclude_cols)

        numerical_features = feature_types["numerical"]
        categorical_features = feature_types["categorical"]
        logger.info(
            f"Feature engineering complete: {len(numerical_features)} numerical, "
            f"{len(categorical_features)} categorical features"
        )

        # Data splitting configuration
        split_method = split_config.method
        test_size = split_config.test_size
        val_size = split_config.val_size
        random_state = split_config.random_state

        # Prepare features (drop metadata columns)
        exclude_cols = metadata_columns + originally_enriched_columns + [target_column]

        # Only exclude date_column if NOT doing temporal split
        if split_method != "temporal":
            exclude_cols = exclude_cols + [date_column]

        X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
        logger.info(f"Features sent to model: {X.columns}")
        logger.info(f"Splitting data using {split_method} method")

        if split_method == "temporal":
            if date_column not in X.columns:
                raise ValueError(
                    f"Date column '{date_column}' required for temporal split"
                )
            X_train, X_val, X_test, y_train, y_val, y_test = temporal_split(
                X, y, date_column=date_column, test_size=test_size, val_size=val_size
            )
            # Drop date column after temporal split
            X_train = X_train.drop(columns=[date_column])
            X_val = X_val.drop(columns=[date_column])
            X_test = X_test.drop(columns=[date_column])
        elif split_method == "stratified":
            X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
                X, y, test_size=test_size, val_size=val_size, random_state=random_state
            )
        else:
            X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
                X, y, test_size=test_size, val_size=val_size, random_state=random_state
            )

        # Extract metadata for splits
        metadata_train = metadata.loc[X_train.index].copy()
        metadata_val = metadata.loc[X_val.index].copy()
        metadata_test = metadata.loc[X_test.index].copy()

        logger.info(
            f"Data splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )

        # Log data split info to MLflow
        tracker.log_params(
            {
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_test": len(X_test),
                "n_features": len(X_train.columns),
            }
        )

        # Identify feature types AFTER splitting and dropping columns
        # This ensures feature lists match what's actually in X_train
        feature_types = identify_feature_types(X_train, exclude_columns=[])
        numerical_features = feature_types["numerical"]
        categorical_features = feature_types["categorical"]

        logger.info(
            f"Feature types identified: {len(numerical_features)} numerical, "
            f"{len(categorical_features)} categorical features"
        )

        # Preprocessing pipeline
        preproc_config = config.preprocessing
        preprocessor = create_preprocessing_pipeline(
            numerical_features=numerical_features,
            categorical_features=categorical_features if categorical_features else None,
            scaling_method=preproc_config.scaling_method,
            imputation_strategy=preproc_config.imputation_strategy,
            handle_outliers=preproc_config.handle_outliers,
        )

        # Train multiple models (including baseline)
        models_to_train = {}
        # model_names = ["baseline", "random_forest", "gradient_boosting"]

        # Train baseline model first
        logger.info(f"\n{'=' * 60}")
        logger.info("Training Baseline Model")
        logger.info(f"{'=' * 60}")

        baseline_model = RecordDifferenceBaseline()
        models_to_train[baseline_model.name] = train_baseline_model(
            baseline_model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        )

        # Log baseline metrics to MLflow
        if (
            baseline_model.name in models_to_train
            and models_to_train[baseline_model.name]
        ):
            baseline_metrics = models_to_train[baseline_model.name].get(
                "test_metrics", {}
            )
            tracker.log_metrics(
                {f"{baseline_model.name}_{k}": v for k, v in baseline_metrics.items()}
            )

        try:
            baseline_model = PointDifferentialBaseline(
                feature_column="pts_diff_avg_L82_delta"
            )

            baseline_model.fit(X_train, y_train)

            # Compute metrics using trainer for consistency
            baseline_trainer = ModelTrainer(
                model=baseline_model,
                task_type="classification",
                random_state=random_state,
            )
            baseline_trainer.is_fitted = True

            baseline_train_metrics = baseline_trainer.evaluate(
                X_train, y_train, prefix="train"
            )
            baseline_val_metrics = baseline_trainer.evaluate(X_val, y_val, prefix="val")
            baseline_test_metrics = baseline_trainer.evaluate(
                X_test, y_test, prefix="test"
            )

            models_to_train[baseline_model.name] = {
                "pipeline": baseline_model,
                "trainer": baseline_trainer,
                "training_results": {
                    "train": baseline_train_metrics,
                    "val": baseline_val_metrics,
                },
                "test_metrics": baseline_test_metrics,
            }

            logger.info(f"Baseline test metrics: {baseline_test_metrics}")
            # Log baseline metrics to MLflow
            tracker.log_metrics(
                {
                    f"{baseline_model.name}_{k}": v
                    for k, v in baseline_test_metrics.items()
                }
            )

        except Exception as e:
            logger.error(
                f"Error training {baseline_model.name_caption}: {e}", exc_info=True
            )

        # Train ML models
        model_names = [model_name for model_name in models_to_train.keys()]
        model_names.extend(["random_forest", "gradient_boosting"])
        for model_name in model_names:
            if "baseline" in model_name:
                continue  # Already trained above
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Training {model_name}")
            logger.info(f"{'=' * 60}")

            try:
                pipeline, trainer, training_results = train_model_with_config(
                    model_name=model_name,
                    model_config=model_config,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    preprocessor=preprocessor,
                    random_state=random_state,
                )

                # Evaluate on test set
                test_metrics = trainer.evaluate(X_test, y_test, prefix="test")

                models_to_train[model_name] = {
                    "pipeline": pipeline,
                    "trainer": trainer,
                    "training_results": training_results,
                    "test_metrics": test_metrics,
                }

                logger.info(f"{model_name} test metrics: {test_metrics}")

                # Log metrics to MLflow
                tracker.log_metrics(
                    {f"{model_name}_{k}": v for k, v in test_metrics.items()}
                )

                # Log model parameters if available
                if hasattr(trainer.model, "named_steps"):
                    model_obj = trainer.model.named_steps.get("model")
                    if model_obj and hasattr(model_obj, "get_params"):
                        model_params = {
                            f"{model_name}_{k}": str(v)
                            for k, v in model_obj.get_params().items()
                        }
                        tracker.log_params(model_params)

            except Exception as e:
                logger.error(f"Error training {model_name}: {e}", exc_info=True)
                continue

        # Compare models
        logger.info(f"\n{'=' * 60}")
        logger.info("Model Comparison - Test Set Results")
        logger.info(f"{'=' * 60}")

        for model_name, model_data in models_to_train.items():
            logger.info(f"\n{model_name}:")
            print_metrics_summary(model_data["test_metrics"], prefix="Test")

        # Find best model
        if models_to_train:
            best_model_name = max(
                models_to_train.keys(),
                key=lambda name: models_to_train[name]["test_metrics"].get(
                    "test_accuracy", 0
                ),
            )
            best_model_data = models_to_train[best_model_name]

            logger.info(f"\n{'=' * 60}")
            logger.info(f"Best Model: {best_model_name}")
            logger.info(
                f"Test Accuracy: {best_model_data['test_metrics']['test_accuracy']:.4f}"
            )
            logger.info(f"{'=' * 60}")

            # Log best model info to MLflow
            tracker.set_tags(
                {
                    "best_model": best_model_name,
                    "best_test_accuracy": str(
                        best_model_data["test_metrics"].get("test_accuracy", 0)
                    ),
                }
            )
            tracker.log_metrics(
                {
                    "best_test_accuracy": best_model_data["test_metrics"].get(
                        "test_accuracy", 0
                    ),
                    "best_test_f1": best_model_data["test_metrics"].get("test_f1", 0),
                    "best_test_precision": best_model_data["test_metrics"].get(
                        "test_precision", 0
                    ),
                    "best_test_recall": best_model_data["test_metrics"].get(
                        "test_recall", 0
                    ),
                }
            )

            best_pipeline = best_model_data["pipeline"]

            # Visualizations
            if eval_config.save_visualizations:
                logger.info("Generating visualizations...")
                output_dir = Path(paths_config.outputs)
                output_dir.mkdir(parents=True, exist_ok=True)

                y_test_pred = best_pipeline.predict(X_test)

            # Confusion matrix
            class_names = sorted(y.unique())
            fig_cm = plot_confusion_matrix(
                y_test,
                y_test_pred,
                class_names=class_names,
                title=f"Test Set Confusion Matrix - {best_model_name}",
            )
            fig_cm.savefig(
                output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight"
            )

            # Feature importance (skip for baseline model)
            # if best_model_name != "baseline":
            if "baseline" not in best_model_name:
                if hasattr(best_pipeline, "named_steps"):
                    trained_model = best_pipeline.named_steps["model"]
                    preprocessor = best_pipeline.named_steps["preprocessor"]
                else:
                    trained_model = best_pipeline
                    preprocessor = None

                if hasattr(trained_model, "feature_importances_"):
                    # Extract actual feature names from the fitted preprocessing pipeline
                    # This accounts for one-hot encoding and other transformations
                    if preprocessor is not None and hasattr(
                        preprocessor, "get_feature_names_out"
                    ):
                        try:
                            # Pass input feature names to get_feature_names_out if needed
                            # ColumnTransformer can infer from transformers, but passing explicitly is safer
                            input_feature_names = list(X_train.columns)
                            feature_names = preprocessor.get_feature_names_out(
                                input_feature_names
                            )
                            # Remove ColumnTransformer prefixes (numerical__, categorical__)
                            feature_names = clean_feature_names(feature_names)
                        except Exception as e:
                            logger.warning(
                                f"Could not extract feature names from preprocessor: {e}. "
                                f"Using original feature names (may cause length mismatch)."
                            )
                            feature_names = numerical_features + (
                                categorical_features or []
                            )
                    else:
                        # Fallback to original feature names if preprocessor not available
                        feature_names = numerical_features + (
                            categorical_features or []
                        )

                    # Validate that feature names length matches feature importances
                    if len(feature_names) != len(trained_model.feature_importances_):
                        logger.error(
                            f"Feature names length ({len(feature_names)}) does not match "
                            f"feature importances length ({len(trained_model.feature_importances_)}). "
                            f"Skipping feature importance plot."
                        )
                    else:
                        fig_importance = plot_feature_importance(
                            trained_model,
                            feature_names,
                            top_n=20,
                            title=f"Feature Importance - {best_model_name}",
                        )
                        fig_importance.savefig(
                            output_dir / "feature_importance.png",
                            dpi=300,
                            bbox_inches="tight",
                        )

                # ROC curve
                y_test_proba = best_pipeline.predict_proba(X_test)
                if y_test_proba.ndim > 1:
                    y_test_proba = y_test_proba[:, 1]
                fig_roc = plot_roc_curve(
                    y_test,
                    y_test_proba,
                    title=f"ROC Curve - {best_model_name}",
                )
                fig_roc.savefig(
                    output_dir / "roc_curve.png", dpi=300, bbox_inches="tight"
                )

                logger.info(f"Visualizations saved to {output_dir}")

                # Log visualizations to MLflow
                tracker.log_artifacts(str(output_dir), artifact_path="visualizations")

            # Save model locally (optional)
            if paths_config.save_local_models:
                logger.info(f"Saving best model ({best_model_name}) locally...")
                registry_path = Path(paths_config.model_registry)
                registry = ModelRegistry(registry_path)

                model_path = registry.save(
                    model=best_pipeline,
                    model_name=f"nba_classification_{best_model_name}",
                    task_type="classification",
                    metrics=best_model_data["test_metrics"],
                    feature_names=list(X_train.columns),
                )

                logger.info(f"Best model saved to: {model_path}")
            else:
                logger.info("Skipping local model save; MLflow handles model logging.")

            # Log best model to MLflow
            tracker.log_model(
                best_pipeline,
                artifact_path="model",
                registered_model_name=f"nba_classification_{best_model_name}",
                input_example=X_train.head(5),
            )

        logger.info("Training complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train classification model")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()

    main(config_path=args.config)
