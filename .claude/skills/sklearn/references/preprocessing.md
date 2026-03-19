# Preprocessing Reference

When to read: modifying preprocessing logic, adding new feature types, or debugging transformation issues.

## ColumnTransformer structure

`create_preprocessing_pipeline()` in `src/ml/features/preprocessing.py` builds:

```
ColumnTransformer
├── "numerical": Pipeline([OutlierHandler?, SimpleImputer, Scaler])
├── "boolean":   Pipeline([FunctionTransformer(bool→int), SimpleImputer(most_frequent)])
├── "categorical": Pipeline([SimpleImputer(most_frequent), OneHotEncoder])
└── remainder="passthrough"
```

## Scaling methods

| Method | Class | When to use |
|---|---|---|
| `"standard"` | `StandardScaler` | Default. Assumes roughly normal distribution |
| `"robust"` | `RobustScaler` | When outliers are present (uses median/IQR) |
| `"minmax"` | `MinMaxScaler` | When bounded [0,1] range is needed |

Configured via `preprocessing.scaling_method` in experiment YAML.

## Boolean feature handling

Boolean columns are separated from numerical because `SimpleImputer` with `mean`/`median` doesn't support boolean dtype. The boolean pipeline:
1. Casts `bool → int` via `FunctionTransformer`
2. Imputes with `most_frequent` strategy

## Categorical feature handling

- `SimpleImputer(strategy="most_frequent")` for missing values
- `OneHotEncoder(drop="if_binary", sparse_output=False, handle_unknown="ignore")`
- `handle_unknown="ignore"` prevents errors from unseen categories at prediction time

## Common mistakes

- **Missing `get_feature_names_out()`**: Custom transformers must implement this or feature name propagation breaks
- **Boolean imputation**: Don't put booleans in the numerical pipeline — `mean` imputation on booleans raises errors
- **Unseen categories**: Without `handle_unknown="ignore"`, OneHotEncoder raises on new categories at prediction time
- **Fitting on full data**: Preprocessor must be fit on train split only, then transform val/test
- **Outlier handler column scope**: `OutlierHandler` auto-selects all numerical columns if `columns=None` — pass explicit columns if you need selective handling
