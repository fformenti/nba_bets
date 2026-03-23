---
name: data-analyst
description: >
  Data analysis expertise for the NBA bets project. Trigger on tasks involving:
  data quality checks, EDA, outlier detection, distribution analysis, correlation
  analysis, training data audit, feature engineering, data leakage detection,
  or any pre-modeling data investigation. Does NOT cover model training/prediction
  (use ml-pipeline skill) or MLflow tracking (use mlflow skill).
---

# Data Analyst

## Workflow

Always follow this sequence when investigating data:

1. **Structured summary report** — Start with a concise findings table before diving into details
2. **Severity classification** — Tag every finding with a severity level
3. **Actionable recommendations** — Each finding includes a fix or next step

## Severity system

Use three levels, formatted consistently:

```
[CRITICAL] What: <description>
           Where: <column/row/file>
           Why: <impact on model or data integrity>
           Fix: <recommended action>

[WARNING]  What: <description>
           Where: <column/row/file>
           Why: <potential impact>
           Fix: <recommended action>

[INFO]     What: <description>
           Where: <column/row/file>
           Note: <context or observation>
```

- **CRITICAL**: Data corruption, leakage, or issues that will produce wrong results
- **WARNING**: Anomalies that may degrade model performance or indicate upstream problems
- **INFO**: Observations worth noting but not actionable blockers

## Behavior

- Ask clarifying questions if dataset context is ambiguous (e.g., "Is this pre-split or post-split data?")
- Output Python/pandas code snippets for fixes unless user requests otherwise
- When auditing data for this project, always consider the NBA domain context (seasons, teams, conferences, game dates)

---

## Section 1 — Data Quality & Consistency

Check for structural anomalies, categorical inconsistencies, malformed entries, and referential integrity issues.

**Quick checks to always run:**
- `df.info()` for dtypes and nulls
- `df.duplicated().sum()` for exact duplicates
- `df.describe()` for obvious range violations
- Unique counts on categorical columns (team IDs, conferences, seasons)

**Detailed reference:** See `references/data-quality-checks.md` for the full checklist including:
- Structural anomalies (mistyped dtypes, unexpected nulls, duplicate rows)
- Categorical consistency (team name misspellings, casing, canonical values)
- Malformed entries (date formats, ID formats)
- Referential integrity (orphaned rows, broken foreign keys between feature tables and games)

---

## Section 2 — Outlier & Anomaly Detection

Identify statistical outliers and distinguish real extreme values from data errors.

**Default approach:**
1. IQR method for initial screening (1.5x IQR for moderate, 3x for extreme)
2. Z-score for normally distributed features
3. Domain-aware thresholds for NBA-specific features (e.g., rest days > 10 is suspicious, distances > 5000 km is unusual)

**Detailed reference:** See `references/outlier-detection.md` for:
- Statistical methods (IQR, Z-score, isolation forest)
- When to use each method
- Distinguishing legitimate outliers vs errors (COVID seasons, expansion teams)
- Categorical distribution anomalies (unbalanced team representation)

---

## Section 3 — Exploratory Data Analysis (EDA)

Run these checks inline (concise enough to not need a reference file):

### Shape & Structure
```python
print(f"Shape: {df.shape}")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(f"Seasons: {df['season'].nunique()} ({df['season'].min()} to {df['season'].max()})")
print(f"Date range: {df['gameDateOnlyStr'].min()} to {df['gameDateOnlyStr'].max()}")
```

### Descriptive Statistics
- `df.describe()` for numerical features — check min/max for impossible values
- `df.select_dtypes(include='object').describe()` for categoricals — check `unique` and `top`
- Per-season game counts: `df.groupby('season').size()` — flag seasons with unusual counts

### Distributions
- Histograms for continuous features (record, pts_diff_avg, distances)
- Value counts for categorical/ordinal features (conferences, seasons)
- Target variable distribution: `df['win_bool'].value_counts(normalize=True)` — check balance

### Cross-tabulation
- Conference filter balance: `pd.crosstab(df['hometeamConference'], df['awayteamConference'])`
- Games per team per season: `df.groupby(['season', 'hometeamId']).size().describe()`

---

## Section 4 — Correlation & Relationships

### Numerical Correlations
- **Pearson** for linear relationships between lag features
- **Spearman** for monotonic relationships (more robust to outliers)
- Flag pairs with |r| > 0.85 as potential multicollinearity candidates

### Categorical Associations
- **Cramer's V** for categorical-to-categorical relationships
- Conference vs win rate cross-tabulation

### Multicollinearity Detection
```python
from sklearn.preprocessing import StandardScaler
from numpy.linalg import cond

X_scaled = StandardScaler().fit_transform(df[numerical_features])
condition_number = cond(X_scaled)
# condition_number > 30 suggests serious multicollinearity
```

### Target Leakage Candidates
- Correlations with target > 0.9 are suspicious — verify the feature is pre-game
- Features that perfectly separate classes warrant investigation
- Cross-reference with `references/data-leakage.md` for the full risk matrix

---

## Section 5 — Training Data Audit

Evaluate data quality from an ML perspective before training.

**Quick checks:**
- Class balance: `df['win_bool'].value_counts(normalize=True)`
- Feature completeness: percentage of non-null values per feature
- Temporal coverage: games per season, no gaps in date ranges
- Minimum games played filter: verify early-season rows are handled

**Detailed reference:** See `references/training-data-audit.md` for:
- Label noise detection (feature-similarity to opposite class)
- Class imbalance strategies (SMOTE, undersampling, class weights)
- Contamination patterns (train/test leakage, features computed from target)

---

## Section 6 — Feature Engineering

All ETL feature code lives in `src/etl/features/`. The ML pipeline (`src/ml/features/engineering.py`) applies delta transformations and conference features on top.

### Adding a new feature

Follow this step-by-step guide:

1. **Create the feature module** in `src/etl/features/` (e.g., `new_feature.py`)
   - Input: games DataFrame (same structure as other feature modules)
   - Output: DataFrame with join keys (`gameId`, `season`, `teamId` or `gameDateOnlyStr`) + new columns
   - Use `.shift(1)` or `.rolling().shift(1)` to prevent data leakage
   - Save output to a CSV path defined in `src/config/paths.py`

2. **Register in aggregator** (`src/etl/features/aggregator.py`)
   - Import the function in the imports section
   - Call it in `create_features_tables()` and save the CSV
   - Add the merge in `merge_features()` using the appropriate join pattern

3. **Add path constant** in `src/config/paths.py`

4. **Configure lags** (if applicable) in the YAML config under `features.lags`, `features.location_lags`, or `features.distances_lags`

5. **Handle delta transformation** (if applicable)
   - Add to the feature list in `src/ml/features/engineering.py` `create_delta_features()`
   - Decide: does a delta (home - away) make sense for this feature?
   - Not all features need deltas — raw features like `days_at_home` are valid on their own

6. **Update `originally_enriched_columns`** in experiment configs if the raw feature should be excluded from the final feature matrix

7. **Test end-to-end**: `make make-features` then `make train`

### Join patterns used in this project

| Pattern | Join keys | When to use |
|---------|-----------|-------------|
| `join_games_and_teams_feature()` | `gameId`, `season`, `teamId` → `hometeamId`/`awayteamId` | Per-team, per-game features (records, pts_diff, distances) |
| Direct merge on `gameDateOnlyStr` | `gameDateOnlyStr` | League-wide daily features (east_vs_west) |
| Merge on `season` + `teamId` | `season`, `teamId` → `hometeamId`/`awayteamId` | Per-team, per-season features (last_season_record) |
| `get_rested_days()` | `gameDateOnlyStr` + `teamId` | Rest days (special handler with home/away suffix logic) |

### Feature catalog & data leakage

- **Full feature catalog**: See `references/feature-catalog.md` for all existing feature modules, their output columns, join keys, and lag configurations
- **Data leakage prevention**: See `references/data-leakage.md` for the leakage risk matrix, verification checklist, and `.shift(1)` requirements

---

## Reference documentation

Detailed reference content lives in `references/`:
- **`references/data-quality-checks.md`** — Structural anomalies, categorical consistency, referential integrity checks
- **`references/outlier-detection.md`** — Statistical methods, domain-aware thresholds, real vs error distinction
- **`references/training-data-audit.md`** — Label noise, class imbalance, contamination patterns
- **`references/feature-catalog.md`** — Full catalog of existing feature modules and output columns
- **`references/data-leakage.md`** — Leakage risk matrix and verification checklist
