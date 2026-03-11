# MLflow Naming Conventions Best Practices

## Overview

This guide explains MLflow naming conventions and best practices for organizing experiments, runs, and registered models.

## The Three-Level Hierarchy

### 1. **Experiments** (Top Level)
**Purpose**: Group related runs together

**Best Practices**:
- Use descriptive, stable names that represent a logical grouping
- Examples: `nba_bets_classification`, `nba_bets_regression`, `nba_bets_hyperparameter_tuning`
- Keep names consistent across team members
- Avoid timestamps or run-specific info in experiment names

**Current Usage**:
```python
experiment_name = "nba_bets_classification"  # ✅ Good
```

### 2. **Runs** (Middle Level)
**Purpose**: Individual training/inference executions

**Best Practices**:
- Include config name to identify what configuration was used
- Include timestamp for uniqueness and chronological ordering
- Optionally include model name for multi-model experiments
- Format: `config_name-timestamp` or `config_name-model_name-timestamp`

**Examples**:
```
my_experiment-20241215-143022          # ✅ Good: config + timestamp
my_experiment-random_forest-20241215   # ✅ Good: config + model + timestamp
run-1                                  # ❌ Bad: not descriptive
```

**Current Implementation**:
```python
run_name = generate_run_name(
    config_path=config_path,
    include_timestamp=True
)
# Result: "my_experiment-20241215-143022"
```

### 3. **Registered Models** (Production Level)
**Purpose**: Production-ready models with versioning

**Best Practices**:
- Use stable, descriptive names (e.g., `nba_classification_random_forest`)
- **Only register models that are production candidates**
- Versions are created automatically when registering the same name
- Use model aliases for production/staging (e.g., `Production`, `Staging`, `Champion`)

**When to Register**:
- ✅ Model passed validation and is production-ready
- ✅ Model is a candidate for A/B testing
- ❌ Don't register every experimental run (use run-based URIs instead)

**Model URIs**:
```python
# Run-based URI (for experimentation)
"runs:/<run_id>/model"

# Registered model URI (for production)
"models:/nba_classification_random_forest/Production"
"models:/nba_classification_random_forest/5"  # Specific version
```

## Current Behavior Explained

When you see:
```
Registered model 'nba_classification_random_forest' already exists. 
Creating a new version of this model...
Created version '5' of model 'nba_classification_random_forest'.
```

**This means**:
- You're registering a model with the same name multiple times
- MLflow is creating version 5 (versions 1-4 already exist)
- This is **normal and expected** for production workflows
- Each version represents a different training run

## Recommendations

### Option 1: Register Only Production Candidates (Recommended)
```python
# In config or as parameter
register_model = False  # Default: don't register experimental runs

# Only register if explicitly enabled
if register_model:
    registered_model_name = f"nba_classification_{best_model_name}"
else:
    registered_model_name = None  # Use run-based URI instead
```

### Option 2: Use Model Aliases for Production
```python
# After registering, promote best version to Production alias
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.set_registered_model_alias(
    name="nba_classification_random_forest",
    alias="Production",
    version=best_version
)
```

### Option 3: Use Tags for Organization
```python
tracker.set_tags({
    "task": "classification",
    "config_file": config_path.stem,
    "experiment_type": "training",
    "model_type": best_model_name,
    "status": "production"  # or "experimental"
})
```

## Workflow Examples

### Experimental Run (Don't Register)
```python
# Training script
run_name = "my_experiment-20241215-143022"
tracker.log_model(model, registered_model_name=None)  # Don't register
# Use: runs:/<run_id>/model
```

### Production Candidate (Register)
```python
# Training script
run_name = "my_experiment-20241215-143022"
tracker.log_model(
    model, 
    registered_model_name="nba_classification_random_forest"
)
# Creates version 5, use: models:/nba_classification_random_forest/5
```

### Production Deployment (Use Alias)
```python
# After validation, promote to Production
client.set_registered_model_alias(
    name="nba_classification_random_forest",
    alias="Production",
    version=5
)
# Use: models:/nba_classification_random_forest/Production
```

## Summary

| Level | Purpose | Naming Convention | Example |
|-------|---------|-------------------|---------|
| **Experiment** | Group related runs | Stable, descriptive | `nba_bets_classification` |
| **Run** | Individual execution | Config + timestamp | `my_experiment-20241215-143022` |
| **Registered Model** | Production models | Stable name, versions auto-created | `nba_classification_random_forest` |

**Key Takeaway**: The versioning message you see is normal. Consider only registering models that are production candidates, and use run-based URIs for experimentation.

