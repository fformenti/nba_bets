---
name: sklearn
description: >
  Scikit-learn expertise grounded in this project's patterns. Trigger on tasks involving:
  sklearn pipelines, ColumnTransformer, custom transformers, preprocessing (scaling/
  imputation/encoding), model selection (GridSearchCV, RandomizedSearchCV), cross-
  validation strategies, writing new sklearn-compatible code, or debugging sklearn API
  usage.
---

# Scikit-learn

## How this project uses sklearn

- **Pipeline structure**: `Pipeline([("preprocessor", ColumnTransformer(...)), ("model", estimator)])`
- **Preprocessing**: `create_preprocessing_pipeline()` in `src/ml/features/preprocessing.py` builds a `ColumnTransformer` with numerical, boolean, and categorical sub-pipelines
- **Custom transformers**: `FeatureSelector`, `MissingValueImputer`, `OutlierHandler` in `preprocessing.py`; `BorutaShapSelector` in `selection.py`; `ThresholdBaseline` and subclasses in `models/baseline.py`
- **Model factory**: `create_model()` in `src/ml/training/model_factory.py` returns `(estimator, param_grid)` pairs

## Key patterns in this codebase

- Step naming: `"preprocessor"` for ColumnTransformer, `"model"` for estimator
- Param grid prefixes: `"model__n_estimators"`, `"model__max_depth"` etc. (Pipeline requires step name prefix)
- Feature name propagation: `ColumnTransformer.set_output(transform="pandas")` → `clean_feature_names()` strips `"numerical__"` / `"boolean__"` prefixes
- Preprocessor fit on train only: `preprocessor.fit(X_train)`, then `preprocessor.transform(X_val)` / `preprocessor.transform(X_test)`
- All models must implement: `fit()`, `predict()`, `predict_proba()`, `score()`

## Reference documentation

Detailed reference content lives in `references/`:
- **`references/pipelines.md`** — Pipeline structure, step naming, param grids, feature name propagation
- **`references/preprocessing.md`** — ColumnTransformer structure, scaling methods, common mistakes
- **`references/model-selection.md`** — Current models, adding new models, hyperparameter tuning, CV strategy
- **`references/custom-transformers.md`** — Existing transformers inventory, template for new ones, rules

For raw sklearn API docs, use the context7 plugin.
