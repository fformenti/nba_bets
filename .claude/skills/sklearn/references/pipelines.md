# Pipelines Reference

When to read: building or modifying sklearn pipelines, debugging feature name issues, or setting up hyperparameter tuning.

## Project pipeline structure

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("preprocessor", create_preprocessing_pipeline(...)),  # ColumnTransformer
    ("model", estimator),                                  # from model_factory
])
```

## Step naming conventions

| Step name | Component | Module |
|---|---|---|
| `"preprocessor"` | `ColumnTransformer` | `src/ml/features/preprocessing.py` |
| `"model"` | Estimator (RF, GB, XGB, LGBM) | `src/ml/training/model_factory.py` |

Access inner components: `pipeline.named_steps["preprocessor"]`, `pipeline.named_steps["model"]`

## Param grid naming for tuning

Pipeline requires `step__param` prefix format:

```python
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [5, 10, 15],
}
```

The model factory auto-prefixes with `"model__"`. Custom grids from config are also auto-prefixed if missing.

## Feature name propagation

1. `ColumnTransformer.set_output(transform="pandas")` preserves DataFrame output
2. ColumnTransformer prepends sub-pipeline names: `"numerical__pts_diff_avg_L7_delta"`, `"boolean__is_back_to_back"`
3. `clean_feature_names()` in `model_factory.py` strips these prefixes: `name.split("__", 1)[1]`

## `set_output(transform="pandas")` implications

- Enables pandas DataFrame flow through the entire pipeline
- Required for `clean_feature_names()` to work (needs string column names)
- Set automatically by `create_preprocessing_pipeline()`
