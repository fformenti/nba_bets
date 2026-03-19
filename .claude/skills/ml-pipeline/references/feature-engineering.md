# Feature Engineering Reference

When to read: designing or adding new features.

## Existing feature types (`src/ml/features/engineering.py`)

- Lag rolling averages: `record_L7_HT`, `pts_diff_avg_L14_VT` — actual lags from `[1, 3, 5, 8, 13, 21, 34, 55, 82]`
- Location-specific: `record_L10_HT_at_home`, `record_L41_VT_on_road`
- Delta features: `record_L7_delta`, `pts_diff_avg_L14_delta` (home − away)
- Distance/travel: `distance_L7_delta`, `rested_days_delta`
- Conference features: depend on `conference_filter` value (`same`/`different`/`all`)

## Lag values from config

```yaml
lags: [1, 3, 5, 8, 13, 21, 34, 55, 82]   # Fibonacci-like rolling windows
location_lags: [10, 41]                    # home/road specific windows
distances_lags: [1, 3, 7, 14]             # travel distance windows
```

## Proposing new features

Think about causal factors that exist *before* game tip-off:
- Back-to-back game flags (`is_back_to_back_HT`, `is_back_to_back_VT`)
- Season stage (early/mid/late, playoff pressure)
- Streak features (current win/loss streak length)
- Rest days differential beyond `rested_days_delta`
- Home/away win rate delta at specific lag windows
- Head-to-head historical record delta

For each proposal, specify: the lag/window, which teams (HT/VT), whether a delta makes sense, and confirm no post-game data is used.
