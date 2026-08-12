# Config Schema Reference

When to read: creating a new experiment YAML or modifying config fields.

## `ExperimentConfig` structure

```yaml
data:
  path: "data/processed/regular_season/games_features.csv"
  target_column: "win_bool"
  date_column: "gameDate"
  drop_na: true

filters:
  start_date: "1980-08-01"
  minimum_games: 30          # min games played by each team before row is included

splitting:
  method: "temporal"         # 'temporal' (default/preferred), 'random', or 'stratified'
  test_size: 0.2
  val_size: 0.2
  random_state: 42

feature_engineering:
  lags: [1, 3, 5, 8, 13, 21, 34, 55, 82]       # rolling window sizes (Fibonacci-like)
  location_lags: [10, 41]
  distances_lags: [1, 3, 7, 14]
  metadata_columns: [...]       # columns kept for bookkeeping, excluded from features
  originally_enriched_columns:  # raw enriched cols that are sources for lag features (excluded)
    - "win_bool", "pts_diff", "total_wins_HT", ...
  exclude_columns: []           # any additional columns to drop from feature matrix

preprocessing:
  scaling_method: "standard"   # 'standard', 'robust', or 'minmax'
  imputation_strategy: "mean"
  handle_outliers: false

model:
  type: "classification"
  name: "random_forest"
  register_model: true          # false → use run-based URI; true → register in MLflow Registry
  hyperparameter_tuning:
    enabled: false
    method: "random"            # 'grid' or 'random'
    n_iter: 20
    cv_folds: 3
    scoring: "accuracy"
  random_forest:
    n_estimators: 300
    max_depth: 15
    min_samples_split: 20
    min_samples_leaf: 10
    max_features: "sqrt"
    class_weight: "balanced"
  gradient_boosting:
    n_estimators: 300
    learning_rate: 0.05
    max_depth: 15
    ...

evaluation:
  save_visualizations: true
  output_dir: "outputs"

paths:
  model_registry: "models"
  outputs: "outputs"
  save_local_models: false
```

> **Pydantic gotcha**: `ExperimentConfig` uses `extra="ignore"` — typos in YAML keys are silently dropped. Always verify key names against `src/ml/config/schema.py`.

## `PredictionConfig` structure

```yaml
input_dir: "data/raw/incremental/upcoming_games"
output_path: "data/predictions/upcoming_games_predictions.csv"
features_path: "data/processed/regular_season/games_features.csv"
feature_config_path: "configs/my_experiment.yaml"

data:
  target_column: "win_bool"
  date_column: "gameDate"

model_uris:                   # keyed by conference filter type
  same: "models:/nba_classification_random_forest_same/1"
  different: "models:/nba_classification_random_forest_different/1"

tracking_uri: null
experiment_name: "nba_bets_predictions"
allow_missing_features: true  # if true, fills missing features with NaN; false raises
max_files: null               # limit number of upcoming-game files to load
```
