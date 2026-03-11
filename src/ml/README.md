# Machine Learning Module

This module provides a comprehensive, production-ready structure for training and evaluating scikit-learn models following best practices.

## Structure

```
src/ml/
├── datasets/       # Data loading and partitioning
├── features/        # Feature engineering pipelines
├── models/          # Model training and persistence
├── evaluation/      # Metrics and visualization
├── config/          # Configuration management
└── scripts/         # Example training scripts
```

## Quick Start

### Regression Example

```python
from src.ml.datasets.loaders import load_features
from src.ml.datasets.splitters import train_val_test_split
from src.ml.models.trainer import ModelTrainer
from sklearn.ensemble import RandomForestRegressor

# Load data
X, y = load_features(target_column="homeScore")

# Split data
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
    X, y, test_size=0.2, val_size=0.2
)

# Train model
trainer = ModelTrainer(RandomForestRegressor(), task_type="regression")
trainer.train(X_train, y_train, X_val, y_val)

# Evaluate
metrics = trainer.evaluate(X_test, y_test)
```

### Classification Example

```python
from src.ml.datasets.loaders import load_features
from src.ml.datasets.splitters import train_val_test_split
from src.ml.models.trainer import ModelTrainer
from sklearn.ensemble import RandomForestClassifier

# Load data
X, y = load_features(target_column="homeWin")

# Split data (stratified)
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
    X, y, test_size=0.2, val_size=0.2, stratify=y
)

# Train model
trainer = ModelTrainer(RandomForestClassifier(), task_type="classification")
trainer.train(X_train, y_train, X_val, y_val)

# Evaluate
metrics = trainer.evaluate(X_test, y_test)
```

## Key Features

### Data Partitioning

- **Random Split**: Standard train/val/test split
- **Temporal Split**: Time-aware splitting for time series data
- **Stratified Split**: Maintains class distribution across splits

### Feature Engineering

- **Preprocessing Pipeline**: Automated handling of numerical and categorical features
- **Missing Value Imputation**: Multiple strategies (mean, median, most_frequent)
- **Outlier Handling**: Clipping or removal based on percentiles
- **Scaling**: Standard, Robust, or MinMax scaling

### Model Training

- **Cross-Validation**: Built-in CV support
- **Hyperparameter Tuning**: Grid search and random search
- **Model Persistence**: Save and load models with metadata

### Evaluation

- **Comprehensive Metrics**: Regression (MSE, RMSE, MAE, R²) and Classification (Accuracy, Precision, Recall, F1, ROC-AUC)
- **Visualizations**: Residual plots, confusion matrices, feature importance, learning curves

## Best Practices

1. **Always use train/val/test splits** - Never evaluate on training data
2. **Use temporal splits for time series** - Maintains temporal order
3. **Stratify splits for classification** - Maintains class distribution
4. **Save models with metadata** - Track model versions and performance
5. **Use pipelines** - Combine preprocessing and modeling for consistency
6. **Cross-validate** - Get robust performance estimates

## Running Training Scripts

### Regression

```bash
uv run python -m src.ml.scripts.train_regression
```

### Classification

```bash
uv run python -m src.ml.scripts.train_classifier --config configs/my_experiment.yaml
```

## Predictions (MLflow)

Run predictions for upcoming games stored in `data/raw/incremental/upcoming_games`:

```bash
uv run python -m src.ml.scripts.predict_upcoming --config configs/predict_upcoming.yaml
```

Predictions are saved to `data/predictions/upcoming_games_predictions.csv`.

## Model Registry

Models are saved to the `models/` directory with:
- Model file (`.joblib`)
- Metadata (`.json`) with metrics, feature names, timestamps

Load a saved model:

```python
from src.ml.models.registry import ModelRegistry

registry = ModelRegistry("models")
model, metadata = registry.load("nba_regression")
```

## Configuration

Use the Pydantic `ExperimentConfig` schema to validate YAML configs:

```python
from pathlib import Path
from src.ml.config import load_experiment_config

config = load_experiment_config(Path("configs/my_experiment.yaml"))
print(config.model_dump())
```
