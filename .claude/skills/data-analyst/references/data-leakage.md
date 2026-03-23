# Data Leakage Reference

When to read: detecting leakage or reviewing a new feature/config for correctness.

## Leakage Risk Matrix

| Risk | Examples | Rule |
|------|----------|------|
| **Target leakage** | `win_bool`, `winner`, `homeScore`, `awayScore` | Always in `originally_enriched_columns` or `metadata_columns` |
| **Score proxies** | `pts_diff`, any post-game aggregate | Always excluded via `originally_enriched_columns` |
| **Temporal leakage** | Any feature computed using future game data | Use only `_L{lag}` lagged features |
| **Conference identity** | `hometeamConference`, `awayteamConference` raw columns | Drop after feature construction |
| **Split leakage** | Fitting scaler/imputer on full dataset before split | Preprocessor always fit on train split only, transform val/test |

## The `.shift(1)` Requirement

Every ETL feature module must use `.shift(1)` (or equivalent) to ensure the current game's result is excluded from its own features.

**Correct pattern:**
```python
# shift(1) ensures we only use data from PRIOR games
games["win_bool_l1"] = games.groupby(["teamId", "season"])["win_bool"].shift(1)
games["record_L7"] = (
    games.groupby(["teamId", "season"])["win_bool_l1"]
    .rolling(window=7, min_periods=1)
    .mean()
    .reset_index(level=[0, 1], drop=True)
)
```

**Wrong pattern (leakage):**
```python
# NO shift — current game's result is included in its own feature
games["record_L7"] = (
    games.groupby(["teamId", "season"])["win_bool"]
    .rolling(window=7, min_periods=1)
    .mean()
    .reset_index(level=[0, 1], drop=True)
)
```

### Modules and their shift patterns

| Module | Column shifted | Method |
|--------|---------------|--------|
| `winning_percentage.py` | `win_bool` → `win_bool_l1` | `.shift(1)` before cumsum/rolling |
| `point_differential.py` | `pts_diff` → `pts_diff_L1` | `.shift(1)` before rolling mean |
| `east_vs_west.py` | cumulative sums | `.shift(1)` on grouped cumsum |
| `rest_days.py` | `rest` indicator | `.shift(1)` on rest flag |
| `distances.py` | N/A — distance is a pre-game fact | No shift needed (location is known before the game) |
| `last_season_record.py` | N/A — uses prior season data | No shift needed (prior season is complete) |

## Cumulative vs Rolling Window Risks

### Cumulative features
- Season-long cumulative stats (e.g., `total_wins`, `total_losses`) grow monotonically
- They can encode "time in season" which may confound with genuine signal
- Use rolling windows instead for features that should capture recent form

### Rolling windows
- Short windows (L1, L3) are volatile — high variance, captures hot streaks
- Long windows (L55, L82) are stable — low variance, captures season-level talent
- The Fibonacci-like lag set `[1, 3, 5, 8, 13, 21, 34, 55, 82]` covers both ends

### Window size and leakage interaction
- A `L1` rolling average with `.shift(1)` = the single previous game's value
- A `L82` rolling average with `.shift(1)` = nearly the entire season's average
- Both are safe as long as `.shift(1)` is applied

## Verification Checklist

When asked to detect or verify leakage:

1. **Config audit**: Scan `originally_enriched_columns` and `metadata_columns` — confirm all post-game signals are listed
2. **Feature name scan**: Check for post-game signals (raw scores, `win_bool`, `pts_diff` without `_L{lag}` suffix)
3. **Preprocessor timing**: Verify preprocessor is fit only on train split (not before `temporal_split()`)
4. **Split integrity**: Confirm `temporal_split()` is used and date ordering is correct (`train_max_date < test_min_date`)
5. **Conference columns**: Confirm raw conference columns are dropped after conference feature construction
6. **New feature review**: For any new feature, trace the computation back to source — does it use any information from the current game or future games?

## Common Leakage Traps in This Project

1. **Prediction pipeline**: `build_features_for_prediction()` concatenates historical + upcoming games, then recomputes feature tables — verify upcoming games have placeholder scores (0) and winner (0) so they don't pollute rolling stats
2. **Conference filter**: Applying conference filter before feature computation could exclude games that should contribute to rolling stats
3. **`originally_enriched_columns` typos**: `ExperimentConfig` uses `extra="ignore"` — if you misspell a column name in `originally_enriched_columns`, it's silently ignored and the column leaks into the feature matrix
4. **Feature alignment**: `_align_features()` in prediction adds missing columns filled with 0 — if a post-game column is missing from the feature list, it gets filled with 0 instead of being excluded
