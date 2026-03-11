# Conference Filter Implementation Guide

## Overview

This document explains the implementation of conference-based model training and how to train 3 different model types based on conference matchups.

## Three Model Types

1. **Same Conference (`conference_filter: "same"`)**
   - Only teams from the same conference playing each other
   - **No conference features** in the model
   - Conference-related columns are dropped

2. **Different Conferences (`conference_filter: "different"`)**
   - Only teams from different conferences playing each other
   - **Features added:**
     - `home_conference_vs_away_conference_record`
     - `games_played_at_home_conference`

3. **All Teams (`conference_filter: "all"`)**
   - All teams regardless of conference
   - **Feature added:**
     - `conference_diff_home_advantage_pct` (equals 0.0 for same conference matchups)

## Implementation Improvements

### 1. Configuration-Based Approach

**Before:** Hardcoded boolean flags
```python
only_different_conferences = True
only_same_conferences = False
within_and_across_conferences = False
```

**After:** Config-based with validation
```yaml
filters:
  conference_filter: "different"  # 'same', 'different', or 'all'
```

### 2. Centralized Feature Engineering

Created `apply_conference_features()` function that:
- Ensures consistent feature engineering across training and prediction
- Handles all 3 conference filter types correctly
- Provides clear logging of which features are created

### 3. MLflow Integration

- Conference filter is logged as a parameter and tag
- Model names include conference filter: `nba_classification_random_forest_different`
- Separate model versions for each conference filter type
- Easy filtering and comparison in MLflow UI

### 4. Training All 3 Model Types

You can now train all 3 model types in one command:

```bash
python -m src.ml.scripts.train_classifier --train-all-conference-types
```

This creates 3 separate MLflow runs, one for each conference filter type.

## Usage Examples

### Training a Single Model

**Option 1: Using config file**
```yaml
# configs/my_experiment.yaml
filters:
  conference_filter: "different"  # or "same" or "all"
```

```bash
python -m src.ml.scripts.train_classifier --config configs/my_experiment.yaml
```

**Option 2: Train all 3 types**
```bash
python -m src.ml.scripts.train_classifier \
  --config configs/my_experiment.yaml \
  --train-all-conference-types
```

### Prediction

The prediction script automatically uses the same conference filter from the experiment config:

```yaml
# configs/predict_upcoming.yaml
feature_config_path: "configs/my_experiment.yaml"  # Uses conference_filter from here
model_uri: "models:/nba_classification_random_forest_different/Production"
```

```bash
python -m src.ml.scripts.predict_classifier --config configs/predict_upcoming.yaml
```

## Feature Engineering Logic

### Same Conference (`"same"`)
```python
# No conference features
# Drops all conference-related columns
```

### Different Conferences (`"different"`)
```python
# Creates:
# - home_conference_vs_away_conference_record
# - games_played_at_home_conference
# Drops intermediate columns (east_record_at_east, etc.)
```

### All Teams (`"all"`)
```python
# Creates:
# - conference_diff_home_advantage_pct
#   * 0.0 for same conference teams
#   * Positive/negative values for different conferences
# Drops intermediate columns
```

## MLflow Model Naming

Models are registered with conference filter in the name:
- `nba_classification_random_forest_same`
- `nba_classification_random_forest_different`
- `nba_classification_random_forest_all`

This allows:
- Separate versioning for each model type
- Easy model selection for predictions
- Clear model organization in MLflow UI

## Key Functions

### `apply_conference_features(df, conference_filter)`
Central function that applies the correct conference features based on filter type.
Used in both training and prediction for consistency.

### `train_single_model(config, config_path, experiment_name, tracker)`
Core training logic extracted to allow training multiple variants.
Called by `main()` for single or multiple model training.

## Best Practices

1. **Always use the same conference filter** in training and prediction configs
2. **Use `--train-all-conference-types`** to compare all 3 model types
3. **Check MLflow tags** to filter runs by conference filter type
4. **Model names include conference filter** - use this when selecting models for prediction

## Verification Checklist

- [ ] Config file has `conference_filter` set correctly
- [ ] Training and prediction use same experiment config
- [ ] Model URI matches the conference filter type
- [ ] Features match expected set for the conference filter type
- [ ] MLflow run has correct tags and parameters logged

