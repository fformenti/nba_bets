# Custom Transformers Reference

When to read: writing a new sklearn-compatible transformer or modifying existing ones.

## Existing custom transformers

| Transformer | Location | Purpose |
|---|---|---|
| `FeatureSelector` | `src/ml/features/preprocessing.py` | Select specific columns from DataFrame |
| `MissingValueImputer` | `src/ml/features/preprocessing.py` | Column-aware imputation (numerical + categorical) |
| `OutlierHandler` | `src/ml/features/preprocessing.py` | Clip or remove outliers via percentile bounds |
| `BorutaShapSelector` | `src/ml/features/selection.py` | Feature selection via Boruta + SHAP importance |
| `ThresholdBaseline` | `src/ml/models/baseline.py` | Base class for threshold-based classifiers |
| `RecordDifferenceBaseline` | `src/ml/models/baseline.py` | Predict based on win record difference |
| `PointDifferentialBaseline` | `src/ml/models/baseline.py` | Predict based on point differential |

## Template for a new transformer

```python
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class MyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, param1: str = "default"):
        self.param1 = param1  # must match __init__ param name exactly

    def fit(self, X: pd.DataFrame, y=None):
        # Compute fitted attributes (suffix with _)
        self.learned_value_ = X.mean()
        self.feature_names_ = X.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()  # always copy
        # Apply transformation using self.learned_value_
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_ if self.feature_names_ is not None else input_features
```

## Rules

- **`__init__` param matching**: Every `__init__` parameter must be stored as `self.param = param` with the exact same name. sklearn's `get_params()` / `set_params()` depend on this.
- **Fitted attributes with `_` suffix**: Attributes learned during `fit()` should end with `_` (e.g., `self.learned_value_`, `self.feature_names_`). This is an sklearn convention for introspection.
- **Always `.copy()`**: Call `X.copy()` in `transform()` to avoid mutating the input DataFrame.
- **`get_feature_names_out()` required**: Must be implemented for feature name propagation through `ColumnTransformer` and `Pipeline`. Return the output column names.
- **Return `self` from `fit()`**: Required for method chaining and pipeline compatibility.
