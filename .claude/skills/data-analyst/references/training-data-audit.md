# Training Data Audit Reference

When to read: evaluating data quality from an ML perspective before training.

## Label Noise Detection

### Feature-Similarity to Opposite Class
Identify samples that look like one class but are labeled as the other.

```python
from sklearn.neighbors import KNeighborsClassifier

# Fit a simple KNN and check for mismatches
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_preds = knn.predict(X_train)

# Samples where KNN disagrees with label — potential noise
disagreements = X_train[knn_preds != y_train]
print(f"Label noise candidates: {len(disagreements)} ({len(disagreements)/len(X_train):.1%})")
```

### Confidence-Based Detection
```python
# Train a model and check prediction confidence on training data
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
probas = model.predict_proba(X_train)

# Low confidence on own label = potential noise
max_proba = probas.max(axis=1)
low_confidence = max_proba < 0.55  # Near coin-flip
print(f"Low-confidence samples: {low_confidence.sum()} ({low_confidence.mean():.1%})")
```

### NBA-Specific Label Considerations
- Home team wins ~55-60% of games historically — significant deviation suggests data issues
- Overtime games have higher upset rates — not label noise
- Early-season games are inherently noisier (small sample for features) — not label noise

## Class Imbalance

### Diagnosis
```python
balance = y_train.value_counts(normalize=True)
print(f"Class balance:\n{balance}")
# For NBA: expect roughly 55-60% home wins, 40-45% away wins
# Imbalance ratio > 2:1 warrants attention
```

### Strategies

| Strategy | When to use | Implementation |
|----------|-------------|----------------|
| **Class weights** | Mild imbalance (< 2:1) | `class_weight='balanced'` in sklearn models |
| **Undersampling** | Large dataset, severe imbalance | `RandomUnderSampler` from imblearn |
| **SMOTE** | Small dataset, need more minority samples | `SMOTE` from imblearn (use with caution on temporal data) |
| **Threshold tuning** | Betting context (different costs for FP vs FN) | Adjust `predict_proba` threshold from 0.5 |

### Important: Temporal Data Constraints
- **Never SMOTE across time boundaries** — synthetic samples could blend future/past information
- Apply resampling **after** temporal split, within the training set only
- Class weights are generally safest for temporal data

## Contamination Patterns

### Train/Test Leakage
```python
# Verify temporal split integrity
train_max_date = X_train['gameDateOnlyStr'].max()
test_min_date = X_test['gameDateOnlyStr'].max()
if train_max_date >= test_min_date:
    print("[CRITICAL] Temporal leakage: train dates overlap with test dates")
```

### Features Computed from Target
Red flags in feature names:
- Any feature containing `win` without a `_L{lag}` suffix
- Raw `pts_diff` without a lag window
- `homeScore`, `awayScore` appearing in the feature matrix
- `winner` column not in `metadata_columns`

### Preprocessor Leakage
```python
# Verify preprocessor was fit on train only
# In this project's pipeline, check that:
# 1. temporal_split() is called BEFORE preprocessor.fit()
# 2. Preprocessor.fit(X_train) then preprocessor.transform(X_test)
# 3. Never preprocessor.fit_transform(X_full) before splitting
```

### Cross-Validation Leakage
- Use `TimeSeriesSplit` for cross-validation, never `KFold` on temporal data
- Ensure feature engineering (rolling averages, lags) is computed before CV, not within each fold
- In this project: features are pre-computed in ETL, so CV leakage risk is lower — but verify the temporal ordering within each fold

## Audit Checklist

1. [ ] Class balance within expected NBA range (55-60% home wins)
2. [ ] No features with > 0.9 correlation to target (potential leakage)
3. [ ] Temporal split verified (no future data in train)
4. [ ] Early-season rows handled (minimum games played filter applied)
5. [ ] No exact duplicate rows in training set
6. [ ] Null percentage per feature is consistent across train/test splits
7. [ ] Feature distributions are similar between train and test (no distribution shift beyond temporal trends)
8. [ ] All post-game columns listed in `originally_enriched_columns` or `metadata_columns`
