# Data Leakage Reference

When to read: detecting leakage or reviewing a new feature/config for correctness.

## Leakage risk matrix

| Risk | Examples | Rule |
|------|----------|------|
| **Target leakage** | `win_bool`, `winner`, `homeScore`, `awayScore` | Always in `originally_enriched_columns` or `metadata_columns` |
| **Score proxies** | `pts_diff`, any post-game aggregate | Always excluded via `originally_enriched_columns` |
| **Temporal leakage** | Any feature computed using future game data | Use only `_L{lag}` lagged features |
| **Conference identity** | `hometeamConference`, `awayteamConference` raw columns | Drop after feature construction |
| **Split leakage** | Fitting scaler/imputer on full dataset before split | Preprocessor always fit on train split only, transform val/test |

## Verification checklist

When asked to detect leakage:
1. Scan `originally_enriched_columns` and `metadata_columns` in the config — confirm all post-game signals are listed
2. Check feature names for post-game signals (e.g., raw scores, `win_bool`, `pts_diff` without `_L{lag}` suffix)
3. Verify preprocessor is fit only on train split (not before `temporal_split()`)
4. Confirm `temporal_split()` is used and date ordering is correct
5. Confirm raw conference columns are dropped after conference feature construction
