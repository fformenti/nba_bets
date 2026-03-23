# Outlier Detection Reference

When to read: identifying statistical outliers and distinguishing real extreme values from data errors.

## Statistical Methods

### IQR Method (Default)
Best for: skewed distributions, initial screening.

```python
def flag_iqr_outliers(series, multiplier=1.5):
    """Flag outliers using IQR method."""
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    return (series < lower) | (series > upper)

# Use 1.5x for moderate outliers, 3x for extreme outliers
moderate = flag_iqr_outliers(df['rested_days_HT'], multiplier=1.5)
extreme = flag_iqr_outliers(df['rested_days_HT'], multiplier=3.0)
```

### Z-Score Method
Best for: approximately normally distributed features.

```python
from scipy import stats

z_scores = stats.zscore(df[numerical_cols].dropna())
outliers = (z_scores.abs() > 3).any(axis=1)
```

### Isolation Forest
Best for: multivariate outlier detection, complex interactions.

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05, random_state=42)
df['is_outlier'] = iso.fit_predict(df[numerical_cols].fillna(0))
# -1 = outlier, 1 = normal
```

## When to Use Each Method

| Method | Best for | Assumptions | Handles multivariate |
|--------|----------|-------------|---------------------|
| IQR | Initial screening, skewed data | None | No |
| Z-score | Normal-ish distributions | Approximate normality | No |
| Isolation Forest | Complex patterns, many features | None | Yes |

## Distinguishing Real Outliers vs Errors

### NBA-Specific Domain Thresholds

| Feature | Normal range | Suspicious | Likely error |
|---------|-------------|------------|--------------|
| `rested_days` | 0-5 | 6-10 | > 14 (except All-Star break) |
| `days_at_home` | 1-15 | 16-25 | > 30 (COVID cap applied in code) |
| `days_on_road` | 1-10 | 11-20 | > 30 (COVID cap applied in code) |
| `distance_L1` | 0-5000 | 5000-8000 | > 10000 |
| `record_L*` | 0.0-1.0 | N/A | < 0 or > 1 |
| `pts_diff_avg_L*` | -20 to +20 | -25 to -20 or +20 to +25 | abs > 30 |
| `games_played` | 0-82 | N/A | > 82 or < 0 |

### Known Legitimate Outliers
- **COVID seasons (2019/20, 2020/21)**: Bubble games, compressed schedules, long rest periods
- **Expansion/relocation**: New teams may have unusual early-season stats
- **All-Star break**: ~7-10 day gap in mid-February
- **Early season**: First few games produce extreme rolling averages due to small sample sizes

### Decision Framework
1. Is the value within the domain's possible range? (If not → error)
2. Is it during a known anomalous period? (COVID, All-Star break → legitimate)
3. Does the outlier appear in multiple related features consistently? (If yes → likely real)
4. Is it an isolated single-row anomaly? (If yes → investigate upstream data)

## Categorical Distribution Anomalies

### Unbalanced Team Representation
```python
# Check games per team per season
games_per_team = pd.concat([
    df.groupby(['season', 'hometeamId']).size(),
    df.groupby(['season', 'awayteamId']).size()
]).groupby(level=[0, 1]).sum()

# Each team plays 82 games; flag significant deviations
print(games_per_team.describe())
```

### Conference Balance
```python
# Check same vs different conference game ratio
same_conf = (df['hometeamConference'] == df['awayteamConference']).sum()
diff_conf = (df['hometeamConference'] != df['awayteamConference']).sum()
print(f"Same conference: {same_conf} ({same_conf/(same_conf+diff_conf):.1%})")
print(f"Different conference: {diff_conf} ({diff_conf/(same_conf+diff_conf):.1%})")
# Expected: ~65% same, ~35% different (NBA scheduling)
```
