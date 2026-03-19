# Model Selection Reference

When to read: adding new models, setting up hyperparameter tuning, or choosing CV strategy.

## Current models in `model_factory.py`

| Name | Class | Key hyperparameters |
|---|---|---|
| `"random_forest"` | `RandomForestClassifier` | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `class_weight` |
| `"gradient_boosting"` | `GradientBoostingClassifier` | `n_estimators`, `learning_rate`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `subsample` |
| `"xgboost"` | `XGBClassifier` | `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` |
| `"lgbm"` | `LGBMClassifier` | `n_estimators`, `learning_rate`, `max_depth`, `num_leaves`, `min_child_samples`, `subsample`, `colsample_bytree` |

## Adding a new sklearn model (4 steps)

1. Add hyperparameter block to YAML config and `ModelConfig` in `src/ml/config/schema.py`
2. Add `elif model_name == "new_model":` block in `create_model()` (`src/ml/training/model_factory.py`)
3. Define default `param_grid` with `"model__"` prefix for each tunable param
4. Add model name to the iteration list in `src/ml/training/experiment.py`

Requirements: must implement `fit()`, `predict()`, `predict_proba()`, `score()`.

## Hyperparameter tuning mechanics

Tuning uses `GridSearchCV` or `RandomizedSearchCV` on the full Pipeline:
- Configured via `model.hyperparameter_tuning` in experiment YAML
- `method: "grid"` or `"random"`, `n_iter`, `cv_folds`, `scoring`
- Param grid comes from `create_model()` or can be overridden via config

## Cross-validation strategy

- **Main evaluation**: Temporal split (never k-fold) — test set chronologically after train
- **Inside tuning only**: k-fold CV is acceptable within `GridSearchCV`/`RandomizedSearchCV` since it operates on the train split only
- Configured via `splitting.method`: `"temporal"`

## Tuning priority order for tree-based models

1. `n_estimators` + `learning_rate` (together, for boosting models)
2. `max_depth` / `num_leaves`
3. `min_child_weight` / `min_child_samples` / `min_samples_split`
4. `subsample` + `colsample_bytree`
5. Regularization (`reg_alpha`, `reg_lambda`) last
