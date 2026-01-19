"""
Example: Using the baseline model for NBA game prediction.

This demonstrates how to use the PointDifferentialBaseline model
which predicts win_bool based on pts_diff_avg_delta.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
from src.ml.models.baseline import PointDifferentialBaseline
from src.ml.models.trainer import ModelTrainer
from src.ml.evaluation.metrics import print_metrics_summary
from src.config import LOCAL_GAMES_FEATURES_PATH

# Load data
print("Loading data...")
df = pd.read_csv(LOCAL_GAMES_FEATURES_PATH)
df = df.dropna()

# Prepare features and target
X = df[["pts_diff_avg_delta"]].copy()  # Only need the feature used by baseline
y = df["win_bool"].copy()

# Split data (simple train/test split for example)
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Create and train baseline model
print("\nTraining baseline model...")
baseline = PointDifferentialBaseline(feature_column="pts_diff_avg_delta")
baseline.fit(X_train, y_train)

# Make predictions
print("\nMaking predictions...")
y_pred = baseline.predict(X_test)

# Evaluate
print("\nEvaluating baseline model...")
trainer = ModelTrainer(baseline, task_type="classification")
trainer.is_fitted = True

test_metrics = trainer.evaluate(X_test, y_test, prefix="test")
print_metrics_summary(test_metrics, prefix="Test")

# Show some example predictions
print("\nExample predictions:")
print("pts_diff_avg_delta >= 0 → Predict win (1)")
print("pts_diff_avg_delta < 0  → Predict loss (0)")
print("\nFirst 10 predictions:")
example_df = pd.DataFrame(
    {
        "pts_diff_avg_delta": X_test["pts_diff_avg_delta"].head(10),
        "predicted": y_pred[:10],
        "actual": y_test.head(10).values,
    }
)
print(example_df)
