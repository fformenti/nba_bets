"""
Example script for training a regression model.

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

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.ml.datasets.loaders import load_features
from src.ml.datasets.splitters import train_val_test_split
from src.ml.features.preprocessing import create_preprocessing_pipeline
from src.ml.models.trainer import ModelTrainer
from src.ml.models.registry import ModelRegistry
from src.ml.evaluation.metrics import print_metrics_summary
from src.ml.evaluation.visualization import plot_residuals, plot_predictions
from src.config import PROJECT_ROOT


def main():
    """Main training function."""

    # ===== Configuration =====
    DATA_PATH = PROJECT_ROOT / "data" / "processed" / "games_features.csv"
    TARGET_COLUMN = "homeScore"  # Example target - adjust based on your needs
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    RANDOM_STATE = 42

    # ===== Load Data =====
    print("Loading data...")
    X, y = load_features(
        file_path=DATA_PATH,
        target_column=TARGET_COLUMN,
        drop_na=True,
    )

    print(f"Loaded {len(X)} samples with {len(X.columns)} features")

    # ===== Split Data =====
    print("\nSplitting data...")
    # Use temporal split if you have a date column, otherwise use random split
    # Option 1: Random split
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_state=RANDOM_STATE,
    )

    # Option 2: Temporal split (uncomment if you have a date column)
    # X_train, X_val, X_test, y_train, y_val, y_test = temporal_split(
    #     X,
    #     y,
    #     date_column="gameDate",  # Adjust column name
    #     test_size=TEST_SIZE,
    #     val_size=VAL_SIZE,
    # )

    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    # ===== Feature Engineering =====
    print("\nSetting up feature engineering pipeline...")

    # Identify numerical and categorical features
    numerical_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Remove non-feature columns (IDs, dates, etc.)
    exclude_cols = ["gameId", "gameDate", "gameDateOnlyStr", "season"]
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

    # ===== Model Selection =====
    print("\nInitializing model...")

    # Option 1: Random Forest
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Option 2: Linear Regression (uncomment to use)
    # model = LinearRegression()

    # Create full pipeline
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # ===== Training =====
    print("\nTraining model...")
    trainer = ModelTrainer(
        model=pipeline,
        task_type="regression",
        random_state=RANDOM_STATE,
    )

    training_results = trainer.train(X_train, y_train, X_val, y_val)

    print("\nTraining Results:")
    print_metrics_summary(training_results["train"], prefix="Train")
    if "val" in training_results:
        print_metrics_summary(training_results["val"], prefix="Validation")

    # ===== Evaluation =====
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(X_test, y_test, prefix="test")
    print_metrics_summary(test_metrics, prefix="Test")

    # ===== Visualization =====
    print("\nGenerating visualizations...")

    # Predictions on test set
    y_test_pred = pipeline.predict(X_test)

    # Plot residuals
    fig_residuals = plot_residuals(y_test, y_test_pred, title="Test Set Residuals")
    fig_residuals.savefig(
        PROJECT_ROOT / "outputs" / "residuals.png", dpi=300, bbox_inches="tight"
    )

    # Plot predictions vs actual
    fig_predictions = plot_predictions(
        y_test, y_test_pred, title="Test Set Predictions"
    )
    fig_predictions.savefig(
        PROJECT_ROOT / "outputs" / "predictions.png", dpi=300, bbox_inches="tight"
    )

    print("Visualizations saved to outputs/")

    # ===== Save Model (optional local registry) =====
    SAVE_LOCAL_MODEL = False
    if SAVE_LOCAL_MODEL:
        print("\nSaving model...")
        registry_path = PROJECT_ROOT / "models"
        registry = ModelRegistry(registry_path)

        model_path = registry.save(
            model=pipeline,
            model_name="nba_regression",
            task_type="regression",
            metrics=test_metrics,
            feature_names=list(X.columns),
        )

        print(f"Model saved to: {model_path}")
    else:
        print("\nSkipping local model save; MLflow should log the model instead.")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
