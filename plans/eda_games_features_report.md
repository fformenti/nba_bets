# EDA Report: games_features.csv

**Dataset**: 36,397 rows × 70 columns | 31 seasons (1995/96 – 2025/26)

**Generated**: 2026-03-23 | Pre-training audit of `data/processed/regular_season/games_features.csv`

---

## [CRITICAL] — Action required before training

### C1. `east_record_adjusted` / `west_record_adjusted` exceed [0, 1] bounds

- **183 rows** where `east_record_adjusted` reaches **2.5**; **132 rows** where `west_record_adjusted` reaches **1.5**
- **Root cause** (`src/etl/features/east_vs_west.py:152-173`): The formula `(east_record / games_played_at_east) * (games_played_east_vs_west / 2.0)` is unbounded. Early in each season when few cross-conference games are played, the ratio blows up.
- **Impact**: These are active feature columns (not excluded). Values > 1.0 create outlier inputs that can distort model training, especially for tree splits.
- **Distribution of out-of-bounds values**:
  - `east_record_adjusted > 1.0`: mean=1.645, range=[1.125, 2.5]
  - `west_record_adjusted > 1.0`: mean=1.290, range=[1.071, 1.5]
- **Recommendation**: Either clip to [0, 1] or fix the adjustment formula. Alternatively, these values self-correct later in the season and the `minimum_games_train=30` filter removes many but not all.

### C2. 6 preseason/international games in dataset

- All from **2025/26** season — opponents: **United, Jerusalem B.C., Loong-Lions, Phoenix (non-NBA)**
- Away team IDs are non-standard (15016, 50014, 50013, 15018), `awayteamConference` and `awayteamLocation` are null, `gameType` is null
- **Detail rows**:
  ```
  gameId    date                  home        away              conf_match  gameType
  12500009  2025-10-03 05:30:00   Pelicans    United           West/NaN    NaN
  12500026  2025-10-04 20:00:00   Nets        Jerusalem B.C.   East/NaN    NaN
  12500011  2025-10-04 23:00:00   Pelicans    Phoenix          West/NaN    NaN
  12500032  2025-10-06 20:00:00   Spurs       Loong-Lions      West/NaN    NaN
  12500043  2025-10-09 22:30:00   Clippers    Loong-Lions      West/NaN    NaN
  12500055  2025-10-13 20:00:00   Timberwolves Loong-Lions     West/NaN    NaN
  ```
- **Impact**: These shouldn't be in `regular_season/games_features.csv`. They corrupt conference features and will have null features.
- **Recommendation**: Filter out rows where `awayteamConference` is null during ingestion, or add `awayteamId` validation.

---

## [WARNING] — Should address, not blocking

### W1. Home win rate drift across temporal splits

| Split | Rows | Date Range | Win Rate | Δ from Train |
|-------|------|-----------|----------|--------------|
| Train (60%) | 21,838 | 1995-11 → 2014-02 | **0.6010** | baseline |
| Val (20%) | 7,279 | 2014-02 → 2020-01 | **0.5818** | -1.92% |
| Test (20%) | 7,280 | 2020-01 → 2026-02 | **0.5501** | -5.09% |

- The model will be trained on a higher home-court advantage era (60.1%) and tested on a lower one (55.0%).
- **Recommendation**: Consider recalibrating predictions or using sample weighting to up-weight recent seasons. The config already has `weighting.enabled: true` with `saturation_K: 30`, which should partially address this.

### W2. `min_games_train=30` filter drops 39% of same-conference data

- Same-conf games total: 23,423
- After min_games filter: **14,349** (loss: **9,074 rows**, 38.7%)
- This filter removes early-season games where lag features are sparse. Trade-off is acceptable, but the effective training set is smaller than raw dataset suggests.

### W3. Multicollinearity: `pts_diff_avg_L34_HT` vs `pts_diff_avg_L10_HT` at r=0.87

- Only one high-correlation pair above 0.85 threshold in the feature set
- **Condition number** of feature matrix: **9.07** (healthy, well below 30 threshold)
- Not severe for tree models, but could cause instability in linear models or inflate feature importance for correlated features

### W4. Extreme `rested_days` values due to COVID break

- `rested_days_HT` max = **145**, `rested_days_VT` max = **144**
- Root: COVID-era break (140-day gap in 2019/20 between 2020-03-11 and 2020-07-30)
- 14,407 HT values flagged by 3×IQR, though most are `rest_days=0` (IQR=0, so threshold is strictly positive)
- Config has `handle_outliers: false` — these will pass through unchanged
- **Impact**: Tree models handle this fine; no action needed

### W5. Stale entries in `train_same.yaml` config

- `conference_diff_east_pct` and `distance` are listed in `originally_enriched_columns` but **don't exist in the CSV**
- Harmless (Pydantic `extra="ignore"` means the pipeline won't crash)
- **Recommendation**: Clean up config — remove these two columns from the list

### W6. HT vs VT cross-correlations for same-league features

| Pair | r | Status |
|------|---|--------|
| `games_played_HT` vs `games_played_VT` | 0.998 | Excluded (metadata) |
| `rested_days_HT` vs `rested_days_VT` | 0.835 | Active feature |
| `total_wins_HT` vs `total_wins_VT` | 0.721 | Excluded (enriched) |
| `total_losses_HT` vs `total_losses_VT` | 0.720 | Excluded (enriched) |

- `rested_days` pair at 0.835 is legitimate — both teams play under the same league schedule
- **No join errors detected**

---

## [INFO] — Expected behavior, no action needed

### Data Integrity

| # | Finding |
|---|---------|
| I1 | Shape confirmed: **36,397 × 70** (33 float64, 22 int64, 14 object, 1 bool) |
| I2 | No exact duplicates, no duplicate `gameId`s |
| I3 | Conferences clean: only **"East"** and **"West"** values (plus 6 null for preseason) |
| I4 | Teams: 30 unique home IDs, 34 away IDs (non-NBA teams in 2025/26) |
| I5 | Label integrity: **100%** — `win_bool == (homeScore > awayScore)` for every row |

### Temporal & Seasonality

| # | Finding |
|---|---------|
| I6 | Short seasons correctly present: 1998/99 (725, lockout), 2011/12 (990, lockout), 2019/20 (1,059, COVID), 2025/26 (897, in progress) |
| I7 | Only 1 gap > 7 days within seasons: COVID break (140 days, 2020-03-11 → 2020-07-30) |
| I8 | No temporal overlap between train/val/test splits |
| I9 | Same-conf home win rate: **58.90%** | Cross-conf: **58.35%** (expected — minimal variation) |
| I10 | Overall home win rate: **58.70%** across all 31 seasons |

### Anomalies

| # | Finding |
|---|---------|
| I11 | Score extremes: homeScore [49, 175], awayScore [53, 176] — all valid |
| I12 | 30 games with \|pts_diff\| > 50: all verifiable blowouts, led by Grizzlies 152-79 vs Thunder (2021) |
| I13 | `distance_L1_VT` has one row with value **-1** — negligible rounding artifact |
| I14 | `gameType` 823 null values — all in metadata_columns, safely excluded from features |

### Feature Engineering

| # | Finding |
|---|---------|
| I15 | All `record_L*` columns within [0, 1] bounds (**except adjusted records**) |
| I16 | 26 numeric feature columns remain after excluding metadata + enriched — all legitimate pre-game signals |
| I17 | Top features by \|correlation\| with win_bool:<br/> • `pts_diff_avg_L34_HT`: r=0.2459<br/> • `pts_diff_avg_L41_HT_at_home`: r=0.2279<br/> • `pts_diff_avg_L10_HT`: r=0.2251<br/> → No leakage detected; all correlations < 0.25 |
| I18 | Distance features (L1, L3, L7, L14 for both HT/VT) all >= 0 miles |
| I19 | Conference and last-season records all in [0, 1] (valid) |

### Data Loss Analysis

| # | Finding |
|---|---------|
| I20 | 2,017 total rows with any null value (~5.5% of dataset) |
| I21 | After config filters (start_date + same_conf + min_games_train=30): **14,349 rows** |
| I22 | After final dropna: **14,062 rows** (287 lost to NaN, 2% of filtered set) |
| I23 | Primary null drivers:<br/> • `record_L41_HT_at_home`: 921 nulls (218 exclusive)<br/> • `record_L41_VT_on_road`: 925 nulls (246 exclusive)<br/> • `record_L1_HT`: 484 nulls<br/> • `record_L1_VT`: 441 nulls<br/> → Early-season games lack sufficient history |

---

## Summary Statistics

### Shape & Composition
- **Total rows**: 36,397 | **Columns**: 70
- **Date range**: 1995-11-03 to 2026-02-19 (30+ years of data)
- **Seasons**: 31 unique (1995/96 through 2025/26)
- **Null rows (any col)**: 2,017 (5.5%)

### Feature Counts After Exclusion
- **Metadata columns**: 23 (gameId, team names, scores, conference, etc.)
- **Originally enriched columns**: 21 (target, derived records, distances, etc.)
- **Active feature columns**: 26 (record lags, pts_diff lags, rested_days, distance lags)

### Training Set (after config filters)
- **Rows**: 14,062 (after dropna)
- **Date range**: 1995-11-03 to 2014-02-11 (19 seasons)
- **Win rate**: ~60% (train split only)

---

## Actionable Checklist

- [ ] **C1**: Fix or clip `east/west_record_adjusted` to [0, 1]
- [ ] **C2**: Filter out the 6 preseason/international games from ingestion
- [ ] **W5**: Remove `conference_diff_east_pct` and `distance` from config `originally_enriched_columns`
- [ ] **W1**: Monitor calibration by era; confirm weighting strategy is working

---

## Conclusion

**Ready for training**: Yes, with caveats.

The dataset is clean, well-structured, and temporally sound. The two critical issues (adjusted records exceeding bounds, preseason games) should be fixed before model deployment, but the data quality is high overall. The feature set is free of leakage and multicollinearity is minimal. The temporal drift in home win rate is a **domain reality** (not a data quality issue) that should be addressed via recalibration or weighting, both of which are already partially in place.
