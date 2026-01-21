# NBA Bets Project Improvements

## Summary of Changes

This document outlines the improvements made to address project structure issues and add MLflow experiment tracking.

## Issues Identified

### 1. Config Confusion
- **Problem**: Two config systems existed:
  - `src/ml/config/config.py` with `MLConfig` dataclass (JSON-only, unused)
  - YAML config loading used in training/pipeline
- **Solution**: Replaced `MLConfig` with a Pydantic `ExperimentConfig` schema that validates YAML configs.

### 2. Project Structure
- **Problem**: Two separate “data” concepts (`data_processing/` for ETL and `ml/data/` for datasets) caused naming collisions and confusion
- **Solution**: Standardized naming with `src/etl/` for ingestion/transforms/features and `src/ml/datasets/` for ML dataset loading/splitting

### 3. Missing Experiment Tracking
- **Problem**: No experiment tracking system in place
- **Solution**: Added MLflow integration

## Changes Made

### 1. Added MLflow Integration ✅

#### New Files Created:
- `src/ml/tracking/__init__.py` - MLflow tracking module exports
- `src/ml/tracking/mlflow_tracker.py` - MLflowTracker class for experiment tracking

#### Features:
- Automatic experiment creation and management
- Parameter logging (config, hyperparameters, data splits)
- Metric logging (train/val/test metrics for all models)
- Model artifact logging
- Visualization artifact logging
- Tagging for best models

#### Usage:
```python
from src.ml.tracking import MLflowTracker

with MLflowTracker(experiment_name="nba_bets_classification") as tracker:
    tracker.log_params({"learning_rate": 0.01})
    tracker.log_metrics({"accuracy": 0.95})
    tracker.log_model(trained_model)
```

### 2. Updated Training Script ✅

- Integrated MLflow tracking into `train_classifier.py`
- All experiments now automatically tracked
- Metrics, parameters, models, and visualizations logged to MLflow
- Best model automatically logged and tagged

### 3. Dependencies ✅

- Added `mlflow>=2.8.0` to `pyproject.toml`

## Recommendations for Future Improvements

### 1. Replace MLConfig with Pydantic
The `MLConfig` class in `src/ml/config/config.py` was unused and JSON-only. It has been replaced with a Pydantic schema (`ExperimentConfig`) that validates the YAML configs used in practice.

### 2. Project Structure Reorganization
Applied a clearer separation of responsibilities:
- `src/etl/` for ingestion, transformation, and feature engineering
- `src/ml/datasets/` for dataset loading and splitting utilities

### 3. Config Management
Current setup uses YAML configs from `configs/` folder and now validates them via Pydantic. Next steps:
- Version configs with experiments (already logged via MLflow)
- Add stricter validation (constraints, enums) as the schema evolves

### 4. MLflow Best Practices
- Set up MLflow server for centralized tracking (optional)
- Use model registry for production models
- Add experiment tags for better organization
- Consider logging feature importance plots

## How to Use MLflow

### View Experiments Locally

1. Start MLflow UI:
```bash
uv run mlflow ui
```

2. Open browser to `http://localhost:5000`

3. View experiments, compare runs, and download models

### Run Training with Tracking

```bash
uv run python -m src.ml.scripts.train_classifier --config configs/my_experiment.yaml
```

All runs are automatically tracked in MLflow!

### Query MLflow Programmatically

```python
import mlflow

# Get experiment
experiment = mlflow.get_experiment_by_name("nba_bets_classification")

# Search runs
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
best_run = runs.loc[runs['metrics.test_accuracy'].idxmax()]

# Load model
model = mlflow.sklearn.load_model(f"runs:/{best_run.run_id}/model")
```

## Next Steps

1. ✅ MLflow integration complete
2. ✅ MLConfig replaced with Pydantic schema
3. ✅ Standardize structure with `src/etl/` and `src/ml/datasets/`
4. ⚠️ Test MLflow tracking with a training run
5. ⚠️ Set up MLflow server if needed for team collaboration

