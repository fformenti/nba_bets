"""
Example script for training a classification model.

This script demonstrates best practices for:
- Data loading and partitioning
- Feature engineering
- Model training
- Model evaluation
- Model persistence
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.ml.data.splitters import temporal_split
from src.ml.features.preprocessing import create_preprocessing_pipeline
from src.ml.models.trainer import ModelTrainer
from src.ml.models.registry import ModelRegistry
from src.ml.evaluation.metrics import print_metrics_summary
from src.ml.evaluation.visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
)
from src.config import PROJECT_ROOT
from src.config import LOCAL_GAMES_FEATURES_PATH
import pandas as pd


def main():
    """Main training function."""

    # ===== Configuration =====
    games: pd.DataFrame = pd.read_csv(LOCAL_GAMES_FEATURES_PATH)
    games_filtered_nulls: pd.DataFrame = games.dropna().copy(deep=True)
    TARGET_COLUMNS = ["win_bool", "pts_diff"]
    TARGET_CLASSIFICATION_COLUMN = "win_bool"

    DATE_COLUMN = "gameDate"
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    RANDOM_STATE = 42

    # Hyperparameter tuning configuration
    USE_HYPERPARAMETER_TUNING = False  # Set to False to use default parameters
    N_ITER = 20  # Number of random search iterations (reduced for faster tuning)
    CV_FOLDS = 3  # Number of cross-validation folds (reduced for faster tuning)
    # Note: 20 iterations × 3 CV folds = 60 model fits
    # For faster tuning, reduce N_ITER to 10-15 or CV_FOLDS to 3
    # For more thorough search, increase N_ITER to 50-100 and CV_FOLDS to 5

    # Conference columns: preserved in metadata for plotting (not used as features)
    # We create conference_diff_east_pct feature instead of one-hot encoding
    CONFERENCE_COLUMNS = ["hometeamConference", "awayteamConference"]

    # Define metadata columns (useful for plotting/inspection/joining later)
    # Note: Conference columns are included here to preserve original string values
    METADATA_COLUMNS = [
        "gameId",  # For joining later
        "winner",
        "hometeamCity",  # For plotting
        "hometeamId",  # For plotting
        "homeScore",
        "awayteamName",  # For plotting
        "awayteamId",  # For plotting
        "awayScore",
        "hometeamConference",  # Original string values for plotting
        "awayteamConference",  # Original string values for plotting
        "winnerteamConference",
        "gameType",
        "gameDateOnlyStr",
        "season",
    ]

    AUXILIARY_COLUMNS = [
        "total_wins_HT",
        "total_losses_HT",
        "total_wins_VT",
        "total_losses_VT",
        "total_wins_HT_at_home",
        "total_losses_HT_at_home",
        "total_wins_VT_on_road",
        "total_losses_VT_on_road",
        "games_played_HT_at_home",
        "games_played_VT_on_road",
        "games_played_HT",
        "games_played_VT",
    ]

    # Comibe features
    # ===== Feature Engineering =====
    print("\nSetting up feature engineering pipeline...")

    games_features = games_filtered_nulls.copy(deep=True)
    games_features["record_delta"] = (
        games_features["record_HT"] - games_features["record_VT"]
    )
    games_features["record_delta_L5"] = (
        games_features["record_L5_HT"] - games_features["record_L5_VT"]
    )
    games_features["record_delta_L13"] = (
        games_features["record_L13_HT"] - games_features["record_L13_VT"]
    )
    games_features["record_delta_L26"] = (
        games_features["record_L26_HT"] - games_features["record_L26_VT"]
    )
    games_features["record_delta_at_location"] = (
        games_features["record_HT_at_home"] - games_features["record_VT_on_road"]
    )
    games_features["record_delta_L5_at_location"] = (
        games_features["record_L5_HT_at_home"] - games_features["record_L5_VT_on_road"]
    )
    games_features["record_delta_L13_at_location"] = (
        games_features["record_L13_HT_at_home"]
        - games_features["record_L13_VT_on_road"]
    )
    games_features["record_delta_L26_at_location"] = (
        games_features["record_L26_HT_at_home"]
        - games_features["record_L26_VT_on_road"]
    )
    games_features["pts_diff_avg_delta"] = (
        games_features["pts_diff_avg_HT"] - games_features["pts_diff_avg_VT"]
    )
    games_features["pts_diff_avg_L5_delta"] = (
        games_features["pts_diff_avg_L5_HT"] - games_features["pts_diff_avg_L5_VT"]
    )
    games_features["pts_diff_avg_L13_delta"] = (
        games_features["pts_diff_avg_L13_HT"] - games_features["pts_diff_avg_L13_VT"]
    )
    games_features["pts_diff_avg_L26_delta"] = (
        games_features["pts_diff_avg_L26_HT"] - games_features["pts_diff_avg_L26_VT"]
    )
    games_features["pts_diff_avg_at_location_delta"] = (
        games_features["pts_diff_avg_HT_at_home"]
        - games_features["pts_diff_avg_VT_on_road"]
    )
    games_features["pts_diff_avg_L5_at_location_delta"] = (
        games_features["pts_diff_avg_L5_HT_at_home"]
        - games_features["pts_diff_avg_L5_VT_on_road"]
    )
    games_features["pts_diff_avg_L13_at_location_delta"] = (
        games_features["pts_diff_avg_L13_HT_at_home"]
        - games_features["pts_diff_avg_L13_VT_on_road"]
    )
    games_features["pts_diff_avg_L26_at_location_delta"] = (
        games_features["pts_diff_avg_L26_HT_at_home"]
        - games_features["pts_diff_avg_L26_VT_on_road"]
    )
    games_features["rested_days_delta"] = (
        games_features["rested_days_HT"] - games_features["rested_days_VT"]
    )
    games_features["days_at_home_delta"] = (
        games_features["days_at_home"] + games_features["days_on_road"]
    )

    # Conference-based feature: difference between binary conference values * east_wins_pct_L1
    # Encode conferences as binary: East=1, West=0
    games_features["hometeamConference_binary"] = games_features[
        "hometeamConference"
    ].map({"East": 1, "West": 0})
    games_features["awayteamConference_binary"] = games_features[
        "awayteamConference"
    ].map({"East": 1, "West": 0})

    # Calculate difference between conference binary values
    games_features["conference_diff"] = (
        games_features["hometeamConference_binary"]
        - games_features["awayteamConference_binary"]
    )

    # Multiply by east_wins_pct_L1 to create the new feature
    games_features["conference_diff_east_pct"] = (
        games_features["conference_diff"] * games_features["east_wins_pct_L1"]
    )

    # Drop the intermediate binary columns (keep original conference columns for one-hot encoding)
    games_features = games_features.drop(
        columns=[
            "hometeamConference_binary",
            "awayteamConference_binary",
            "conference_diff",
        ]
    )

    # Drop original features
    drop_original_features = [
        "record_HT",
        "record_VT",
        "record_L5_HT",
        "record_L5_VT",
        "record_L13_HT",
        "record_L13_VT",
        "record_L26_HT",
        "record_L26_VT",
        "record_HT_at_home",
        "record_VT_on_road",
        "record_L5_HT_at_home",
        "record_L5_VT_on_road",
        "record_L13_HT_at_home",
        "record_L13_VT_on_road",
        "record_L26_HT_at_home",
        "record_L26_VT_on_road",
        "pts_diff_avg_HT",
        "pts_diff_avg_VT",
        "pts_diff_avg_L5_HT",
        "pts_diff_avg_L5_VT",
        "pts_diff_avg_L13_HT",
        "pts_diff_avg_L13_VT",
        "pts_diff_avg_L26_HT",
        "pts_diff_avg_L26_VT",
        "pts_diff_avg_HT_at_home",
        "pts_diff_avg_VT_on_road",
        "pts_diff_avg_L5_HT_at_home",
        "pts_diff_avg_L5_VT_on_road",
        "pts_diff_avg_L13_HT_at_home",
        "pts_diff_avg_L13_VT_on_road",
        "pts_diff_avg_L26_HT_at_home",
        "pts_diff_avg_L26_VT_on_road",
        "rested_days_HT",
        "rested_days_VT",
        "days_at_home",
        "days_on_road",
        "east_wins_pct_L1",
    ]
    games_features = games_features.drop(columns=drop_original_features)

    print(f"features after engineering: {games_features.columns}")

    # Prepare target
    y: pd.Series = games_features[TARGET_CLASSIFICATION_COLUMN]

    # Extract metadata before creating features (keep for later use)
    metadata: pd.DataFrame = games_features[METADATA_COLUMNS].copy()

    # Columns to exclude from features (target + metadata including conference columns)
    # Conference columns are excluded from features since we use conference_diff_east_pct instead
    columns_to_drop = (
        TARGET_COLUMNS
        + AUXILIARY_COLUMNS
        + METADATA_COLUMNS  # Includes conference columns - they're only for metadata/plotting
    )
    # Keep date column temporarily for temporal splitting
    X_with_date: pd.DataFrame = games_features.drop(columns=columns_to_drop)

    print(f"Loaded {len(X_with_date)} samples with {len(X_with_date.columns)} features")
    print(f"Target distribution:\n{y.value_counts()}")

    # ===== Split Data =====
    print("\nSplitting data...")
    # Use temporal split if you have a date column, otherwise use random split
    # Option 1: Random split
    # X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
    #     X_with_date.drop(columns=[DATE_COLUMN]),
    #     y,
    #     test_size=TEST_SIZE,
    #     val_size=VAL_SIZE,
    #     random_state=RANDOM_STATE,
    #     stratify=y,  # Stratify by target for balanced splits
    # )

    # Option 2: Temporal split (using date column)
    X_train, X_val, X_test, y_train, y_val, y_test = temporal_split(
        X_with_date,
        y,
        date_column=DATE_COLUMN,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
    )

    print(f"features before splitting: {X_train.columns}")

    # Extract metadata for each split (using indices from splits)
    metadata_train = metadata.loc[X_train.index].copy()
    metadata_val = metadata.loc[X_val.index].copy()
    metadata_test = metadata.loc[X_test.index].copy()

    # Drop date column from feature sets after splitting
    X_train = X_train.drop(columns=[DATE_COLUMN])
    X_val = X_val.drop(columns=[DATE_COLUMN])
    X_test = X_test.drop(columns=[DATE_COLUMN])

    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Note: Metadata columns are stored in metadata_train, metadata_val, metadata_test for later use.
    # Conference columns (hometeamConference, awayteamConference) are:
    #   - Excluded from features (we use conference_diff_east_pct numerical feature instead)
    #   - Preserved in metadata as original strings ("East"/"West") for plotting
    # You can join metadata back with predictions for plotting/inspection:
    # results_df = pd.concat([
    #     metadata_test.reset_index(drop=True),
    #     pd.DataFrame({'prediction': y_test_pred, 'actual': y_test}).reset_index(drop=True)
    # ], axis=1)

    # Identify numerical and categorical features
    numerical_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Remove non-feature columns (IDs, dates, metadata, etc.)
    # Conference columns are excluded - we use conference_diff_east_pct feature instead
    exclude_cols = METADATA_COLUMNS + ["gameDate"]

    numerical_features = [col for col in numerical_features if col not in exclude_cols]
    categorical_features = [
        col for col in categorical_features if col not in exclude_cols
    ]

    print(f"Numerical features: {len(numerical_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(
        numerical_features=numerical_features,
        categorical_features=categorical_features if categorical_features else None,
        scaling_method="standard",
        imputation_strategy="mean",
        handle_outliers=True,
    )

    # ===== Model Training and Comparison =====
    print("\n" + "=" * 60)
    print("Training Multiple Models for Comparison")
    print("=" * 60)

    models_to_train = {}
    results = {}

    # ===== Random Forest =====
    print("\n[Model 1/2] Random Forest Classifier")
    print("-" * 60)

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    rf_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", rf_model),
        ]
    )

    # Hyperparameter tuning for Random Forest
    if USE_HYPERPARAMETER_TUNING:
        import time

        total_fits = N_ITER * CV_FOLDS
        print(
            f"Performing hyperparameter tuning ({N_ITER} iterations × {CV_FOLDS} CV folds)..."
        )
        start_time = time.time()

        rf_param_grid = {
            "model__n_estimators": [50, 100, 200, 300],
            "model__max_depth": [5, 7, 10, 15, None],
            "model__min_samples_split": [10, 20, 30, 50],
            "model__min_samples_leaf": [5, 10, 15, 20],
            "model__max_features": ["sqrt", "log2", 0.5, 0.7],
            "model__class_weight": ["balanced", "balanced_subsample", None],
        }

        rf_trainer = ModelTrainer(
            model=rf_pipeline,
            task_type="classification",
            random_state=RANDOM_STATE,
        )

        rf_trainer.hyperparameter_tuning(
            X_train=X_train,
            y_train=y_train,
            param_grid=rf_param_grid,
            method="random",
            cv=CV_FOLDS,
            n_iter=N_ITER,
            scoring="f1",
        )

        elapsed_time = time.time() - start_time
        print(f"Tuning completed in {elapsed_time / 60:.1f} minutes")
        if hasattr(rf_trainer, "search_results_"):
            best_params = rf_trainer.search_results_.best_params_
            best_score = rf_trainer.search_results_.best_score_
            print(f"Best CV F1 Score: {best_score:.4f}")

        rf_pipeline = rf_trainer.model
        rf_trainer = ModelTrainer(
            model=rf_pipeline,
            task_type="classification",
            random_state=RANDOM_STATE,
        )
    else:
        rf_trainer = ModelTrainer(
            model=rf_pipeline,
            task_type="classification",
            random_state=RANDOM_STATE,
        )

    # Train Random Forest
    print("Training Random Forest...")
    rf_training_results = rf_trainer.train(X_train, y_train, X_val, y_val)
    rf_test_metrics = rf_trainer.evaluate(X_test, y_test, prefix="test")

    models_to_train["Random Forest"] = {
        "pipeline": rf_pipeline,
        "trainer": rf_trainer,
        "training_results": rf_training_results,
        "test_metrics": rf_test_metrics,
    }

    # ===== Gradient Boosting =====
    print("\n[Model 2/2] Gradient Boosting Classifier")
    print("-" * 60)

    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
    )

    gb_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", gb_model),
        ]
    )

    # Hyperparameter tuning for Gradient Boosting
    if USE_HYPERPARAMETER_TUNING:
        print(
            f"Performing hyperparameter tuning ({N_ITER} iterations × {CV_FOLDS} CV folds)..."
        )
        start_time = time.time()

        gb_param_grid = {
            "model__n_estimators": [50, 100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__max_depth": [3, 5, 7, 10],
            "model__min_samples_split": [10, 20, 30, 50],
            "model__min_samples_leaf": [5, 10, 15, 20],
            "model__subsample": [0.8, 0.9, 1.0],
        }

        gb_trainer = ModelTrainer(
            model=gb_pipeline,
            task_type="classification",
            random_state=RANDOM_STATE,
        )

        gb_trainer.hyperparameter_tuning(
            X_train=X_train,
            y_train=y_train,
            param_grid=gb_param_grid,
            method="random",
            cv=CV_FOLDS,
            n_iter=N_ITER,
            scoring="f1",
        )

        elapsed_time = time.time() - start_time
        print(f"Tuning completed in {elapsed_time / 60:.1f} minutes")
        if hasattr(gb_trainer, "search_results_"):
            best_params = gb_trainer.search_results_.best_params_
            best_score = gb_trainer.search_results_.best_score_
            print(f"Best CV F1 Score: {best_score:.4f}")

        gb_pipeline = gb_trainer.model
        gb_trainer = ModelTrainer(
            model=gb_pipeline,
            task_type="classification",
            random_state=RANDOM_STATE,
        )
    else:
        gb_trainer = ModelTrainer(
            model=gb_pipeline,
            task_type="classification",
            random_state=RANDOM_STATE,
        )

    # Train Gradient Boosting
    print("Training Gradient Boosting...")
    gb_training_results = gb_trainer.train(X_train, y_train, X_val, y_val)
    gb_test_metrics = gb_trainer.evaluate(X_test, y_test, prefix="test")

    models_to_train["Gradient Boosting"] = {
        "pipeline": gb_pipeline,
        "trainer": gb_trainer,
        "training_results": gb_training_results,
        "test_metrics": gb_test_metrics,
    }

    # ===== Compare Models =====
    print("\n" + "=" * 60)
    print("Model Comparison - Test Set Results")
    print("=" * 60)

    for model_name, model_data in models_to_train.items():
        print(f"\n{model_name}:")
        print_metrics_summary(model_data["test_metrics"], prefix="Test")

    # Find best model based on test accuracy
    best_model_name = max(
        models_to_train.keys(),
        key=lambda name: models_to_train[name]["test_metrics"].get("test_accuracy", 0),
    )
    best_model_data = models_to_train[best_model_name]

    print("\n" + "=" * 60)
    print(f"Best Model: {best_model_name}")
    print(f"Test Accuracy: {best_model_data['test_metrics']['test_accuracy']:.4f}")
    print("=" * 60)

    # ===== Visualization =====
    print("\nGenerating visualizations for best model...")

    # Use best model for visualization
    best_pipeline = best_model_data["pipeline"]
    y_test_pred = best_pipeline.predict(X_test)

    # Plot confusion matrix
    class_names = sorted(y.unique())
    fig_cm = plot_confusion_matrix(
        y_test,
        y_test_pred,
        class_names=class_names,
        title=f"Test Set Confusion Matrix - {best_model_name}",
    )
    fig_cm.savefig(
        PROJECT_ROOT / "outputs" / "confusion_matrix.png", dpi=300, bbox_inches="tight"
    )

    # Plot feature importance (if model supports it)
    trained_model = best_pipeline.named_steps["model"]
    if hasattr(trained_model, "feature_importances_"):
        feature_names = numerical_features + (categorical_features or [])
        fig_importance = plot_feature_importance(
            trained_model,
            feature_names,
            top_n=20,
            title=f"Feature Importance - {best_model_name}",
        )
        fig_importance.savefig(
            PROJECT_ROOT / "outputs" / "feature_importance.png",
            dpi=300,
            bbox_inches="tight",
        )

    # Plot ROC curve
    # Get predicted probabilities for positive class
    y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    fig_roc = plot_roc_curve(
        y_test,
        y_test_proba,
        title=f"ROC Curve - {best_model_name}",
    )
    fig_roc.savefig(
        PROJECT_ROOT / "outputs" / "roc_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    print("Visualizations saved to outputs/")

    # ===== Save Best Model =====
    print(f"\nSaving best model ({best_model_name})...")
    registry_path = PROJECT_ROOT / "models"
    registry = ModelRegistry(registry_path)

    model_path = registry.save(
        model=best_pipeline,
        model_name=f"nba_classification_{best_model_name.lower().replace(' ', '_')}",
        task_type="classification",
        metrics=best_model_data["test_metrics"],
        feature_names=list(X_train.columns),
    )

    print(f"Best model saved to: {model_path}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
