# Data Quality Checks Reference

When to read: auditing a dataset for structural issues, inconsistencies, or integrity problems.

## Structural Anomalies

### Dtype Mismatches
- Dates stored as strings instead of datetime (`gameDate`, `gameDateOnlyStr`)
- Numeric IDs stored as float instead of int (common after merges with NaN rows)
- Boolean columns stored as int or object

```python
# Quick dtype audit
for col in df.columns:
    if df[col].dtype == 'object':
        # Check if it should be numeric
        try:
            pd.to_numeric(df[col], errors='raise')
            print(f"[WARNING] {col}: stored as object but looks numeric")
        except (ValueError, TypeError):
            pass
```

### Unexpected Nulls
- Columns that should never be null: `gameId`, `season`, `hometeamId`, `awayteamId`, `gameDate`
- Columns expected to have nulls early in season: lag features (first N games have insufficient history)
- Null patterns that indicate merge failures: all features null for specific teams or dates

```python
# Null audit
null_pct = df.isnull().mean().sort_values(ascending=False)
suspicious = null_pct[(null_pct > 0) & (null_pct < 1.0)]
print("Columns with partial nulls (potential merge issues):")
print(suspicious)
```

### Duplicate Rows
- Exact duplicates: `df.duplicated().sum()`
- Duplicate game IDs: `df['gameId'].duplicated().sum()` — should always be 0 for games_features
- Duplicate team-date pairs: can indicate double-counting in feature tables

```python
# Duplicate audit
print(f"Exact duplicates: {df.duplicated().sum()}")
print(f"Duplicate gameIds: {df['gameId'].duplicated().sum()}")
print(f"Duplicate home team-date: {df.duplicated(subset=['gameDateOnlyStr', 'hometeamId']).sum()}")
```

## Categorical Consistency

### Team Names & IDs
- Team IDs should be consistent integers across all seasons
- Team names may change (e.g., relocation, rebranding) — check `src/config/constants.py` for canonical mappings
- Verify all team IDs in feature tables exist in the games table

```python
# Team consistency check
games_teams = set(df['hometeamId'].unique()) | set(df['awayteamId'].unique())
print(f"Unique teams: {len(games_teams)}")
# NBA has 30 teams; more or fewer is suspicious
```

### Conference Values
- Should only be `"East"` or `"West"` — check for case variations, whitespace, or nulls
- Verify conference assignments are consistent per team within a season

```python
for col in ['hometeamConference', 'awayteamConference']:
    if col in df.columns:
        print(f"{col} values: {df[col].unique()}")
```

### Season Format
- Expected format: `"YYYY/YY"` (e.g., `"2024/25"`)
- Verify continuity: no missing seasons in the range

## Malformed Entries

### Date Formats
- `gameDateOnlyStr` should match `YYYY-MM-DD` pattern
- `gameDate` should be parseable as datetime
- Dates should fall within NBA regular season windows (roughly October to April)

```python
# Date validation
dates = pd.to_datetime(df['gameDateOnlyStr'], errors='coerce')
invalid_dates = dates.isna().sum()
if invalid_dates > 0:
    print(f"[CRITICAL] {invalid_dates} unparseable dates")

# Season window check
months = dates.dt.month
off_season = ((months >= 5) & (months <= 9)).sum()
if off_season > 0:
    print(f"[WARNING] {off_season} games in off-season months (May-Sep)")
```

### Score Validation
- `homeScore` and `awayScore` should be positive integers
- No tied scores (NBA games cannot end in a tie)
- Reasonable range: typically 70-160 per team

## Referential Integrity

### Feature Table → Games
- Every row in a feature table should correspond to a valid `gameId` in the games table
- After merging, check for rows where all feature columns are null (broken join)

### Cross-Table Consistency
- Team IDs in feature tables should match team IDs in games
- Date strings in feature tables should match dates in games
- Season values should be consistent across all tables

```python
# Post-merge integrity check
feature_cols = [c for c in df.columns if '_L' in c or '_delta' in c]
all_null_rows = df[feature_cols].isnull().all(axis=1)
if all_null_rows.sum() > 0:
    print(f"[CRITICAL] {all_null_rows.sum()} rows with ALL features null — likely broken merge")
    # Inspect which teams/dates are affected
    print(df.loc[all_null_rows, ['gameId', 'gameDateOnlyStr', 'hometeamId', 'awayteamId']].head())
```
