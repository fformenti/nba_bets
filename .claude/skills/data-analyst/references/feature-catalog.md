# Feature Catalog Reference

When to read: understanding existing features, proposing new ones, or debugging feature engineering.

## ETL Feature Modules (`src/etl/features/`)

### `winning_percentage.py`
Computes team win rates with rolling windows.

| Function | Output columns | Join keys | Suffixes |
|----------|---------------|-----------|----------|
| `calculate_record()` | `record_L{lag}`, `total_wins`, `total_losses`, `games_played` | `gameId`, `season`, `teamId` | `_HT`, `_VT` |
| `calculate_home_record()` | Same as above (home games only) | `gameId`, `season`, `teamId` | `_HT_at_home` |
| `calculate_away_record()` | Same as above (away games only) | `gameId`, `season`, `teamId` | `_VT_on_road` |

**Leakage prevention**: Uses `.shift(1)` on `win_bool` before computing rolling averages.

### `point_differential.py`
Computes rolling average point differential.

| Function | Output columns | Join keys | Suffixes |
|----------|---------------|-----------|----------|
| `calculate_pts_diff()` | `pts_diff_avg_L{lag}` | `gameId`, `season`, `teamId` | `_HT`, `_VT` |
| `calculate_home_pts_diff()` | Same (home games only) | `gameId`, `season`, `teamId` | `_HT_at_home` |
| `calculate_away_pts_diff()` | Same (away games only, sign flipped) | `gameId`, `season`, `teamId` | `_VT_on_road` |

**Leakage prevention**: Uses `.shift(1)` on `pts_diff` before computing rolling averages.

### `east_vs_west.py`
Computes cumulative conference vs conference records at the league level.

| Function | Output columns | Join keys |
|----------|---------------|-----------|
| `make_east_west_record(games)` | `east_record_adjusted`, `west_record_adjusted`, `games_played_east_vs_west` | `gameDateOnlyStr` |
| `make_east_west_record(games, location="East")` | `east_record_at_east`, `games_played_at_east` | `gameDateOnlyStr` |
| `make_east_west_record(games, location="West")` | `west_record_at_west`, `games_played_at_west` | `gameDateOnlyStr` |

**Leakage prevention**: Cumulative sums use `.shift(1)` so current day's results are excluded.

### `rest_days.py`
Computes rest days between games and consecutive days at home/on road.

| Function | Output columns | Join keys | Suffixes |
|----------|---------------|-----------|----------|
| `make_rested_days_table()` | `rested_days`, `days_at_home`, `days_on_road` | `gameDateOnlyStr`, `teamId` | `_HT`, `_VT` |

**Special handling**: `days_at_home` and `days_on_road` are capped at 30 to handle COVID-era outliers. These features are **not** delta-transformed — `days_at_home` and `days_on_road` are used as raw features, while `rested_days` gets a delta.

### `distances.py`
Computes rolling average travel distance based on game locations.

| Function | Output columns | Join keys | Suffixes |
|----------|---------------|-----------|----------|
| `make_teams_distances_table_season()` | `distance_L{lag}` | `gameId`, `season`, `teamId`, `gameDateOnlyStr` | `_HT`, `_VT` |

**Note**: Uses driving distances from `LOCATIONS_DISTANCES_PATH`. The `distance_L1` represents the current game's travel, not a lagged value — the rolling window includes the current day.

### `last_season_record.py`
Computes each team's previous-season win percentage.

| Function | Output columns | Join keys | Suffixes |
|----------|---------------|-----------|----------|
| `create_last_season_record()` | `last_season_record` | `season`, `teamId` | `_HT`, `_VT` |
| `create_last_season_home_record()` | `last_season_record` | `season`, `teamId` | `_HT_at_home` |
| `create_last_season_away_record()` | `last_season_record` | `season`, `teamId` | `_VT_on_road` |

**Note**: Expansion teams are filled with the minimum win percentage of all teams in the previous season. First season in dataset returns NaN (handled by ML pipeline's mean imputer).

## ML Feature Transformations (`src/ml/features/engineering.py`)

These run **after** ETL features are merged, during training/prediction.

### Delta Features (`create_delta_features()`)
Computes `home - away` differences. The raw home/away columns are dropped after delta creation.

**Features that get deltas:**
- `record_L{lag}_delta` (for each lag in `features.lags`)
- `pts_diff_avg_L{lag}_delta` (for each lag in `features.lags`)
- `distance_L{lag}_delta` (for each lag in `features.distances_lags`)
- `rested_days_delta`
- `last_season_record_delta`

**Location-specific deltas:**
- `record_L{lag}_at_location_delta` (for each lag in `features.location_lags`)
- `pts_diff_avg_L{lag}_at_location_delta` (for each lag in `features.location_lags`)
- `last_season_record_at_location_delta`

**Special case:**
- `days_at_home_delta` = `days_at_home + days_on_road` (sum, not difference)

### Conference Features (`apply_conference_features()`)
Applied based on `conference_filter` value:

| Filter | Function called | Output columns |
|--------|----------------|----------------|
| `same` | None (drops conference columns) | — |
| `different` | `get_home_conference_vs_away_conference_record()` | `home_conference_vs_away_conference_record`, `games_played_at_home_conference` |
| `all` | `create_conference_delta()` | `conference_diff_home_advantage_pct` |

## Lag Configuration

```yaml
# From experiment YAML configs
features:
  lags: [1, 3, 5, 8, 13, 21, 34, 55, 82]    # Fibonacci-like rolling windows
  location_lags: [10, 41]                      # Home/road specific windows
  distances_lags: [1, 3, 7, 14]               # Travel distance windows
```

## Proposing New Features

Think about causal factors that exist *before* game tip-off:
- Back-to-back game flags (`is_back_to_back_HT`, `is_back_to_back_VT`)
- Season stage (early/mid/late, playoff pressure)
- Streak features (current win/loss streak length)
- Rest days differential beyond `rested_days_delta`
- Home/away win rate delta at specific lag windows
- Head-to-head historical record delta

For each proposal, specify:
1. The lag/window (if applicable)
2. Which teams (HT/VT or both)
3. Whether a delta makes sense
4. Confirm no post-game data is used
5. The join pattern (see SKILL.md Section 6)
