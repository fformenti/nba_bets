---
name: ml-pipeline
description: >
  Project-specific ML pipeline expertise for the NBA bets project. Trigger on tasks
  involving: training scripts, prediction scripts, adding models, experiment setup,
  model evaluation, data preparation, code organization, or any part of the
  training/prediction lifecycle. Does NOT cover MLflow tracking details (use mlflow
  skill), raw sklearn API questions (use sklearn skill), or feature engineering and
  data leakage (use data-analyst skill).
---

# ML Pipeline

## Project conventions

- **Python runner**: Always use `uv run python -m ...` (never `python` or `pip`)
- **Config-driven**: All experiments use Pydantic-validated YAML configs (`ExperimentConfig`, `PredictionConfig`)
- **One model, every game**: training produces a single model. The same/cross-conference split it replaced measured worse (+0.0047 log loss) — see docs/CONFERENCE_SPLIT.md. The conference *signal* is kept as two features, both 0.0 for same-conference games.
- **Temporal split**: Default and preferred split. Test set must always be chronologically after training data
- **MLflow**: All experiments tracked via `MLflowTracker` context manager → see mlflow skill for details

## Architecture at a glance

```
src/ml/
├── config/        # Pydantic schemas + YAML loader
├── datasets/      # load_features(), temporal_split(), stratified_split()
├── evaluation/    # compute_classification_metrics(), visualizations
├── features/      # engineering.py (delta/lag/conference features), preprocessing.py
├── models/        # ModelTrainer, baseline models, ModelRegistry
├── prediction/    # io.py (load_upcoming_games())
├── scripts/       # train_classifier.py, predict_classifier.py (entry points)
├── tracking/      # MLflowTracker
└── training/      # experiment.py, data_prep.py, model_factory.py, runners.py
```

### `training/` sub-module responsibilities

- `experiment.py` — top-level orchestration: `train_single_model()`, `generate_run_name()`
- `data_prep.py` — data loading/filtering: `load_and_validate_data()`, `prepare_data()`, `filter_minimum_games_played()`
- `model_factory.py` — model instantiation: `create_model()`, `clean_feature_names()`
- `runners.py` — training loops: `train_baseline_model()`, `train_model_with_config()`

## Adding a new model

1. Add hyperparameter dict block in YAML config and to `ModelConfig` in `src/ml/config/schema.py`
2. Add model instantiation in `src/ml/training/model_factory.py` `create_model()` function
3. Extend the model list in `src/ml/training/experiment.py` to include the new model name

Scikit-learn API required: `fit()`, `predict()`, `predict_proba()`, `score()`.

## Training flow summary

`train_classifier.py` trains one model:
1. `load_and_validate_data()` → filter by `min_season` + minimum games → `prepare_data()`
2. `create_delta_features()` + `create_conference_features()`
3. Temporal/stratified/random split → preprocessor fit on train only
4. Train `RecordDifferenceBaseline`, `PointDifferentialBaseline`, `random_forest`, `gradient_boosting`
5. Best model selected by `test_accuracy` from `test_metrics`
6. Saves visualizations, optionally registers model (→ mlflow skill for registry details)
7. Metrics stored as `{model_name}_test_accuracy`, `{model_name}_test_f1`, etc.

## Prediction flow summary

`predict_classifier.py` scores the whole slate with the one deployed model:
1. `load_upcoming_games()` from `data/raw/incremental/upcoming_games/`
2. `fix_upcoming_games_cols()` — adds placeholder post-game columns (scores=0, winner=0)
3. `build_features_for_prediction()` — concatenates historical + upcoming, recomputes feature tables via `create_features_tables()`, then calls `merge_features()` for upcoming rows only
4. `prepare_features_for_model()` — drops metadata, enriched, and excluded columns
5. `_align_features()` — reorders/fills missing columns to match training feature set
6. Output: `upsert_predictions()` writes **one row per game**, keyed on `PREDICTION_KEY = ["gameId"]` — re-predicting a slate replaces rows rather than appending. Columns include `prediction`, `home_win_probability`

## Organizing ML code

Follow this module boundary contract:
- `scripts/` — thin entry points only; delegate to `training/` and `prediction/`
- `training/` — orchestration (`experiment.py`), data prep (`data_prep.py`), model factory (`model_factory.py`), training runners (`runners.py`)
- `models/` — model definitions, trainer, registry; no data loading or feature logic
- `features/` — pure transformations; no model logic or I/O
- `datasets/` — loading and splitting only; no feature transforms
- `tracking/` — MLflow only; no business logic
- `prediction/` — prediction I/O helpers (e.g., `load_upcoming_games()`)

## End-to-end lifecycle checklist

1. [ ] Config YAML created with correct `originally_enriched_columns` and `metadata_columns`
2. [ ] Features built with `make make-features`
3. [ ] Temporal split confirmed (no future data in train)
4. [ ] Preprocessor fit on train split only
5. [ ] Model trained and logged (→ mlflow skill for tracking details)
6. [ ] Baseline comparison included
7. [ ] `register_model: true` and model registered
8. [ ] Prediction config updated with the new `model_uri` (`make train PROMOTE=1`)
9. [ ] Prediction script tested end-to-end with `make predict-upcoming`

## Common pitfalls in this codebase

- **Feature alignment**: `predict_classifier.py` aligns prediction features to training feature set via `_align_features()`; new features must appear consistently in both `games_features.csv` and the upcoming game feature construction
- **Off-population features**: a feature whose estimand is only defined on a sub-population (the two conference features) must be imputed at its neutral value elsewhere — 0 for a signed difference, 0.5 for a raw rate — never NaN (`drop_na: true` would delete those rows) and never the population mean.
- **Config schema silent failures**: `ExperimentConfig` uses `extra="ignore"` — typos in YAML keys are silently ignored; double-check key names against `schema.py`
- **Prediction output upserts**: re-predicting a slate replaces its rows. `src/betting/bets.py` raises on a duplicate gameId rather than sizing a full budget against each of two disagreeing probabilities.
- **`originally_enriched_columns` vs `exclude_columns`**: The former are raw source columns used to compute lag features — they must be excluded from the feature matrix. The latter are any ad-hoc extra columns to drop.

## Reference documentation

Detailed reference content lives in `references/`:
- **`references/config-schema.md`** — Full config YAML examples with all fields. Read when creating or modifying configs.

For feature engineering and data leakage detection, use the **data-analyst** skill.
