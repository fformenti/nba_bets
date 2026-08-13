# Feature Catalog Reference

When to read: understanding existing features, proposing new ones, debugging feature
engineering, or deciding whether a column that exists is actually reaching a model.

Last full audit: 2026-08-12. Every column between raw ETL output and a trained model
was traced; the ledger at the bottom is the result.

---

## 0. How to read this file

### The four classifications

Every column in `data/processed/regular_season/games_features.csv` is one of:

| Class | Meaning |
|-------|---------|
| **active** | Reaches a model as a feature (possibly after delta transformation). |
| **precursor** | Exists only so another feature can be computed. Fine — the ledger names what it feeds. |
| **metadata** | Identity / plotting / masking column from raw data. Never a model input, by design. |
| **barren** | Computed, not a precursor, not metadata, consumed by nothing. This is the bug class. |

"Precursor" is a claim about *where* the consumption happens. Most of this pipeline's
precursors are consumed **inside the ETL**, before `merge_features` runs — `sos_L{lag}`
feeds `sos_adj_record` from the in-memory table, not from the merged `sos_L21_HT` column.
That distinction matters: the merged copy can be barren while the feature it "feeds" is
active. Several were.

### Two configs, two jobs

- **`configs/features.yaml`** — which feature *tables* the ETL builds. Only `lags`,
  `location_lags`, `sos_adj_alpha`, `gds_beta`. Nothing here decides what a model
  consumes.
- **`configs/train/_defaults.yaml`** — which columns a *model* consumes:
  `feature_engineering.features.<group>.{enabled, delta}`, plus lag overrides where the
  model wants a different window set than the ETL builds.

`features.yaml` used to carry an `enabled` key on every group, documented as "ML only".
It was read by nothing — `aggregator.py` only unpacks the lag fields, and `_defaults.yaml`
merges afterwards and re-declares `enabled` for every group. Five of the thirteen asserted
the opposite of what shipped (`record: enabled: true` while no sklearn model had seen
`record`). The keys were removed in the 2026-08-12 audit. **If you find yourself writing
`enabled` in `features.yaml`, you want `_defaults.yaml`.**

### Nothing is selected by hand

`feature_selection.enabled: true` in `_defaults.yaml` runs Boruta-SHAP
(`src/ml/features/selection.py`) on `X_train` at the start of every training run:
20 iterations, each fitting LightGBM on the real features plus a shuffled shadow copy of
each, counting how often a feature's mean |SHAP| beats the best shadow, then a two-tailed
binomial test at α=0.05.

- `hits > 10` and `p < 0.05` → **confirmed**
- `hits ≤ 10` and `p < 0.05` → **rejected**
- otherwise → **tentative**

`include_tentative: true`, so the model trains on confirmed + tentative. **Rejected
features are dropped from the model even though the YAML enables them.** A group being
`enabled: true` means "offer it to Boruta", not "the model uses it".

Correlated features swap in and out of "confirmed" together. Any single Boruta artifact is
evidence about *that feature pool*, not about a feature in isolation.

**And selection is not a safety net for redundancy.** Two features at r ≈ 1 both get
confirmed — see §3, which is the section to read before enabling any group. Leaving
everything enabled and letting Boruta sort it out is not a safe default; it is how this
pipeline shipped a duplicated feature family.

### Where the Boruta artifacts live

| Path | Feature pool | Status |
|------|--------------|--------|
| `outputs/xgboost/feature_selection/` | current shipping set | authoritative for the deployed model |
| `outputs/feature_audit/feature_selection/` | every group enabled, matched lags (pass B) | authoritative for "has Boruta ever seen X" |
| `outputs/all_models/`, `outputs/all/` | byte-identical to `outputs/xgboost` at time of audit | redundant copies |
| `outputs/same/`, `outputs/different/` | **stale** — conference-split era | ignore; they score `home_conference_vs_away_conference_record` and `games_played_at_home_conference`, columns that no longer exist |
| `mlruns/*/*/artifacts/outputs/feature_selection/` | per-run copy | matches whatever config that run used |

`configs/train/llm_features.yaml` has **no** Boruta artifact and by design never will:
nothing trains from it. `build_llm_dataset` reads it for column selection and splits, then
serializes rows to text. Its extra columns (`streak`, raw `days_at_home`/`days_on_road`) are
therefore unfiltered by feature selection — which is why the redundancy rule in §3 binds
*harder* there than on the sklearn path, not more loosely. A duplicated column on the
sklearn path at least meets a shadow test on the way in; here it goes straight into the
prompt.

---

## 1. ETL feature modules (`src/etl/features/`)

Each module returns a team-level table keyed on `(gameId, season, teamId)` (a few key on
`(season, gameDateOnlyStr)` or `(gameDateOnlyStr, teamId)` instead). `aggregator.py`
merges them onto the games frame twice — once on `hometeamId` with suffix `_HT`, once on
`awayteamId` with suffix `_VT` — plus location-scoped tables with `_HT_at_home` /
`_VT_on_road`.

### `winning_percentage.py`

Rolling win rate. The base quantity most other features refine.

| Function | Output columns | Join keys | Merge suffix |
|----------|---------------|-----------|--------------|
| `calculate_record()` | `record_L{lag}`, `total_wins`, `total_losses`, `games_played` | `gameId`, `season`, `teamId` | `_HT`, `_VT` |
| `calculate_home_record()` | same, home games only | `gameId`, `season`, `teamId` | `_HT_at_home` |
| `calculate_away_record()` | same, road games only | `gameId`, `season`, `teamId` | `_VT_on_road` |

**Leakage**: `.shift(1)` on `win_bool` within `(teamId, season)` before every rolling mean.
`min_periods=1`, so a team's second game already has `record_L82`, computed off one game.
The first game of each season is NaN — this is what `prepare_data`'s `dropna()` removes.

**Note on `calculate_away_record`**: tests `winner == awayteamId` rather than
`winner != hometeamId`. Equivalent for played games, but an unknown winner (the `0`
placeholder for upcoming games) must read as a loss on *both* sides, not a fabricated road
win.

### `point_differential.py`

Two families in one module, and the distinction matters.

| Function | Output columns | Merge suffix |
|----------|---------------|--------------|
| `calculate_pts_diff()` | `pts_diff_avg_L{lag}` | `_HT`, `_VT` |
| `calculate_home_pts_diff()` / `calculate_away_pts_diff()` | same (away sign-flipped) | `_HT_at_home`, `_VT_on_road` |
| `calculate_norm_pts_diff()` | `norm_pts_diff_avg_L{lag}` | `_HT`, `_VT` |
| `calculate_norm_home_pts_diff()` / `calculate_norm_away_pts_diff()` | same | `_HT_at_home`, `_VT_on_road` |
| `_add_season_rolling_avg_total_pts()` | `season_avg_total_pts` (internal) | — |

**Normalization**: `norm_pts_diff = pts_diff / season_avg_total_pts`, where the divisor is
the season-to-date mean of combined points per game, `.shift(1)`-ed. This neutralises era
scoring pace, so a +8 margin in 1998 and a +8 in 2024 are not treated as the same
achievement. The first game of a season falls back to the *previous* season's mean rather
than an all-time mean — mixing the 1940s and 2020s would defeat the purpose and peek
forward.

**Only the normalized family ships — the two are near-duplicates, at r = 0.998.**
`season_avg_total_pts` is an expanding within-season mean, so it is near-constant after the
first weeks of a season; that makes `norm_pts_diff_avg_LX ≈ pts_diff_avg_LX / c_season`, an
affine transform. The entire information difference is between-season rescaling, which is
exactly what the feature exists to do. `norm_point_differential` is the training-facing
member — see the redundancy map in §3.

> This entry briefly claimed the opposite. On 2026-08-12 a Boruta pass confirmed raw L13,
> L34 and L82 alongside their normalized twins, which was read as "not redundant after all"
> and both families were enabled. That is the correlated-pair artifact described in §3: at
> r = 0.998 the two split the SHAP credit and each still beats a shadow. Test accuracy fell
> 0.66pp with both in. Reverted.

**Leakage**: `.shift(1)` on `pts_diff` per `(teamId, season)`. `.fillna(0.0)` on the result,
so unlike `record` these columns have **no NaN** — a first-of-season row reads 0.0, the
correct neutral value for a signed difference.

### `east_vs_west.py`

Cumulative conference-vs-conference records at league level, over interconference games only.

| Call | Output columns | Join keys |
|------|---------------|-----------|
| `make_east_west_record(games)` | `east_record_adjusted`, `west_record_adjusted` | `season`, `gameDateOnlyStr` |
| `make_east_west_record(games, location="East")` | `east_record_at_east` | `season`, `gameDateOnlyStr` |
| `make_east_west_record(games, location="West")` | `west_record_at_west` | `season`, `gameDateOnlyStr` |

`location=None` averages the two venues out; the `location=` variants return exactly the
venue component that average discards. The `games_played_*` columns are divisors inside the
builder and are not returned.

**The join needs `season`**: the right side is one row per (season, date), so joining on
date alone fans the games frame out for any date shared by two seasons. `aggregator.py`
passes `validate="m:1"`.

**Leakage**: cumulative sums `.shift(1)` so the current day's results are excluded.

All four outputs are **precursors** — `create_conference_features` consumes and drops them.

### `rest_days.py`

| Function | Output columns | Join keys | Merge suffix |
|----------|---------------|-----------|--------------|
| `make_rested_days_table()` | `rested_days`, `back_to_back`, `days_at_home`, `days_on_road` | `gameDateOnlyStr`, `teamId` | `_HT`, `_VT` (see below) |

Built on a full team × calendar-day grid per season, not on game rows — that is how
"consecutive days at home" is even definable.

- `rested_days` — days since the previous game.
- `back_to_back` — consecutive days *immediately before* this one on which the team also
  played. `0` = rested yesterday, `1` = back-to-back, `2` = three games in three days.
  Not a boolean.
- `days_at_home` / `days_on_road` — length of the current homestand / road trip. Capped at
  `COVID_BREAK_THRESHOLD = 30`: `days_at_home` clamps to the season's largest non-outlier
  value (keeping the ordering monotone), `days_on_road` resets to 1 (teams went home during
  the break, so the first game back is day one of a new trip).

**Asymmetric merge** (`aggregator.get_rested_days`): `rested_days` and `back_to_back` are
merged for both teams and suffixed. `days_at_home` is merged **only for the home team** and
`days_on_road` **only for the away team**, both unsuffixed. That is deliberate — a
homestand helps the host, a long road trip hurts the visitor, and the reverse pairings are
not interesting. It is also why these two are handled by the `home_and_road` group rather
than by the generic HT/VT delta machinery.

A game with no grid match means the grid has a hole; `get_rested_days` logs a warning
before filling 0, because a silent 0 fabricates a back-to-back.

### `distances.py`

| Function | Output columns | Join keys | Merge suffix |
|----------|---------------|-----------|--------------|
| `make_teams_distances_table_season()` | `distance_L{lag}` | `gameId`, `season`, `teamId`, `gameDateOnlyStr` | `_HT`, `_VT` |

Driving distance between consecutive game cities, from `LOCATIONS_DISTANCES_PATH` (a
tracked, non-regenerable reference file — see CLAUDE.md).

**The one deliberate exception to the `.shift(1)` rule.** `distance_L1` is the *current*
game's travel, not a lagged value: travel into a game is fixed by the schedule and fully
known before tip-off, so including the current row is information the model genuinely has
at prediction time. Documented as such in the source; do not "fix" it.

An unmatched city pair raises rather than filling 0, because 0 miles is indistinguishable
from "stayed home". A *null* location (team-season the lookup does not cover) is warned and
zeroed instead — it cannot be judged either way.

### `strength_of_schedule.py`

| Function | Output columns | Join keys | Merge suffix |
|----------|---------------|-----------|--------------|
| `calculate_strength_of_schedule()` | `sos_L{lag}` | `gameId`, `season`, `teamId`, `gameDate` | `_HT`, `_VT` |

Rolling mean win percentage of the last N opponents, each evaluated as of the current
game's date.

**Leakage, subtle and important**: opponent strength is read at **day** granularity, not
tip-off timestamp. Two games on the same evening tip off hours apart, so a timestamp
comparison let the later one read the earlier one's result — a result the prediction
pipeline cannot have, since it predicts the whole slate before any of it is played. That
was train/serve skew, and the fix is `astype("datetime64[D]")` on both sides of the
`searchsorted`.

An opponent with no completed games this season falls back to their prior-season record,
then to `NEW_FRANCHISE_STRENGTH = 0.200`. Without it, early-season SOS collapses for whole
slates.

`min_opponents` is `3` by default but `aggregator.py` calls it with `1`.

### `sos_adjusted_record.py`

| Function | Output columns | Merge suffix |
|----------|---------------|--------------|
| `calculate_sos_adjusted_record()` | `sos_adj_record_L{lag}` | `_HT`, `_VT`, `_HT_at_home`, `_VT_on_road` |

`sos_adj = raw_win_pct * (team_sos / league_avg_sos) ** alpha`, `alpha = sos_adj_alpha = 1.0`.
Beat a hard schedule and your record scales up; feast on a soft one and it scales down.

`league_avg_sos` is a **season-to-date expanding** mean, not a same-day one: the NBA
regularly plays a single game on a date, so a daily league average would be drawn from two
teams. `game_day` normalizes `gameDate` first, because grouping on the raw timestamp
averages over the ~2 teams sharing an exact tip-off rather than the day's slate.

Output lags are the **intersection** of `record_lags` and `sos_lags`. `aggregator.py`
computes SOS at `set(sos_lags) | set(sos_adj_location_lags)` so the location windows
(5/10/20/41) have SOS available, then saves only the declared `sos_lags` to the SOS table —
that is why `sos_L10` exists inside the build but never reaches the CSV.

### `game_difficulty_score.py`

| Function | Output columns | Merge suffix |
|----------|---------------|--------------|
| `calculate_game_difficulty_score()` | `gds_L{lag}` | `_HT`, `_VT` |
| `calculate_home_gds()` / `calculate_away_gds()` | same, location-scoped | `_HT_at_home`, `_VT_on_road` |

Per-game quality score, then rolled:

```
Win:  GDS = +Q(opponent) * (1 + β * is_away)
Loss: GDS = -(1 - Q(opponent)) * (1 + β * is_home)
```

`β = gds_beta = 0.10`. `Q` is the opponent's SOS-adjusted record at the largest available
lag, falling back to cumulative win% → opponent's prior season → `NEW_FRANCHISE_STRENGTH`,
then clipped to [0, 1].

**What it measures that nothing else does**: win percentage counts a win over the worst team
in the league the same as one over the best. SOS describes who you played without reference
to how it went. GDS is the credit for the specific results, opponent-weighted and
venue-adjusted — beating a good team on the road is the maximum score, losing at home to a
bad team the minimum.

**Leakage**: `gds_raw` is `.shift(1)`-ed per `(teamId, season)` before rolling.

### `streaks.py`

| Function | Output columns | Merge suffix |
|----------|---------------|--------------|
| `calculate_streak()` | `streak` | `_HT`, `_VT` |

Signed consecutive-result run entering the game: `+3` = three straight wins, `-2` = two
straight losses, `0` = first game of the season.

Per-group results are collected as index-aligned Series rather than a flat list — a
positional assignment would silently attach streaks to the wrong rows if the sort keys and
the groupby ever stopped agreeing.

### `playoff_standings.py`

| Function | Output columns | Merge suffix |
|----------|---------------|--------------|
| `compute_playoff_flags()` | `conf_rank`, `games_behind_leader`, `games_behind_above`, `games_ahead_of_below`, `clinched_playoff_berth`, `eliminated_from_playoffs`, `clinched_final_seed`, `indifference_flag` | `_HT`, `_VT` |

Conference standings and playoff status for every (team, game). The most complex builder in
the ETL, and — until this audit — the one whose output was least reachable.

**Standings snapshot definition**: for a game on date D, standings use post-game stats for
teams whose last game was strictly before D, and **pre-game** stats for teams playing on D.
Same day-granularity discipline as `strength_of_schedule`.

**Tiebreakers** (`break_ties`): head-to-head within the tied group → conference win % →
point differential. For 3+ teams still tied after H2H the cascade continues without
re-evaluating H2H within sub-groups — documented simplification.

**Clinch logic is conservative**: `clinched_playoff_berth` requires fewer than `cutoff`
other teams to be *mathematically* able to finish above the team's current win total,
ignoring tiebreakers. Cutoff is 8 pre-2020, 10 after (play-in era).

**`games_remaining` reads the parsed league schedule**, not the games present. Counting
present games is right only once a season is over; mid-season it is badly wrong — a 2025/26
frame stopping in January holds ~690 games, reads as a 46-game season, and teams start
clinching berths in December. `season_lengths_from_schedule()` returns `{}` when the
schedule has not been parsed (`full-rebuild` does not run `parse-league-schedule`), and
callers fall back to counting.

`indifference_flag = eliminated_from_playoffs | clinched_final_seed` — a team with nothing
left to play for. Note `create_delta_features` **inverts** this one:
`indifference_flag_delta = VT - HT`, so a positive value means the *visitor* is the one
coasting, which favours the home team like every other feature's positive direction.

Team-games with a null conference are skipped by the `groupby` and get NaN standings; the
builder logs the count rather than ranking them as a pseudo-conference.

### `teams_arena.py`

| Function | Output columns |
|----------|---------------|
| `build_teams_arena()` | `team_id`, `season`, `home_arena_ids` (internal lookup) |
| `add_neutral_court()` | `neutral_court` (0/1, game-level, no HT/VT split) |

A game is neutral when its `arenaId` is not among the home team's known home arenas
(≥20% of that season's home games — the threshold accommodates split seasons like the
post-Katrina 2005 Hornets). Falls back to the label-based `is_neutral_court_game` when
`arenaId` is null.

### `last_season_record.py`

Six public builders, same shape: aggregate a season win percentage over some scope
(all / home / road), optionally SOS-adjust it, then attach each team's *previous* season
value.

| Function | Output column | Merge suffix |
|----------|--------------|--------------|
| `create_last_season_record()` | `last_season_record` | `_HT`, `_VT` |
| `create_last_season_home_record()` | `last_season_record` | `_HT_at_home` |
| `create_last_season_away_record()` | `last_season_record` | `_VT_on_road` |
| `create_adjusted_last_season_record()` | `adjusted_last_season_record` | `_HT`, `_VT` |
| `create_adjusted_last_season_home_record()` | `adjusted_last_season_record` | `_HT_at_home` |
| `create_adjusted_last_season_away_record()` | `adjusted_last_season_record` | `_VT_on_road` |

The adjusted variants apply the same multiplicative SOS correction as `sos_adj_record`,
using each team's end-of-season `sos_L82`.

**Two different expansion-team fallbacks, on purpose:**
- `_prior_season_lookup` (the six builders above) fills a team with no prior season using
  the **minimum** prior-season win percentage among teams playing that season.
- `build_prior_season_strength` (the opponent-quality prior for SOS and GDS) uses a
  **fixed** `NEW_FRANCHISE_STRENGTH = 0.200`.

Teams in the first season of the dataset stay NaN either way; the ML imputer handles them.

`prev_season()` is public because the prediction pipeline needs it: last-season features are
a lookup into the prior season, so filtering history to the slate's own season makes them
all NaN.

---

## 2. ML transforms (`src/ml/features/engineering.py`)

These run **after** the ETL tables are merged, inside `build_splits` (training) and
`prepare_features_for_model` (prediction). Both call the same functions — that is the
train/serve parity guarantee.

### `create_delta_features()`

For each **enabled** group with `delta: true`, builds `{prefix}_delta = {prefix}_HT - {prefix}_VT`
and drops both sources. Groups in `LOCATION_VARIANT_GROUPS` additionally build
`{prefix}_at_location_delta = {prefix}_HT_at_home - {prefix}_VT_on_road`.

For each **disabled** group, the HT/VT columns are dropped without a delta. This is why
`enabled: false` leaves no trace in the frame.

Three special cases:

| Case | Behaviour | Why |
|------|-----------|-----|
| `indifference_flag` | delta is `VT - HT`, inverted | positive should favour the home team, and it is the *visitor* having nothing to play for that helps |
| `home_and_road` | delta is a **sum**: `days_at_home + days_on_road` | both push the same direction — a long homestand helps the host, a long road trip hurts the visitor — so summing collapses two columns into one without cancelling |
| `neutral_court` | game-level scalar, no HT/VT split, no delta | one value per game, not per team |

Boolean columns are cast to int before subtraction (`_numeric`) — numpy refuses `-` on
bool, which would otherwise make `clinched_playoff_berth_delta` raise instead of producing
the -1/0/1 that "one team clinched and the other didn't" should be.

### `create_conference_features()`

Two features, built for every game. There is no conference filter — the same/cross-conference
model split measured worse and was removed (see `docs/CONFERENCE_SPLIT.md`).

| Column | Meaning | Same-conference value |
|--------|---------|----------------------|
| `conference_diff_home_advantage_pct` | venue-balanced East/West gap, signed by who hosts | `0.0` |
| `conference_home_court_advantage_pct` | home conference's win rate hosting the other conference, minus 0.5 | `0.0` |

Both estimate a quantity defined only on interconference games. Within a conference both
teams come from the same pool, so the effect cancels *exactly* — `0.0` is a known truth, not
an imputation guess. The second must be gated: its ingredients are conference-season-level
and defined on every date, so an ungated lookup hands every East home team one value and
every West home team another, which is a confound rather than noise. Centering on 0.5 is
what makes "no effect" and "no data" the same number.

The east/west inputs (`east_record_adjusted`, `west_record_adjusted`, `east_record_at_east`,
`west_record_at_west`) are consumed and dropped — none may survive as a feature.

> **Worth knowing**: both conference features have been **rejected by Boruta in every run
> examined** (importance 0.006 / 0.005, 0 hits). They are the two features that justified
> collapsing three conference-routed models into one, and feature selection drops them
> before training. The architecture decision still stands on its own backtest, but "the
> conference signal stayed" is currently not true of the shipped model. Flagged, not
> changed — see §6.

### `create_momentum_features()`

Replaces a correlated `[short, long]` lag-delta pair with
`{feature}_momentum_L{short}_L{long}_delta = {short}_delta - {long}_delta`, dropping both
sources. Driven by `feature_engineering.momentum_pairs`. **No shipped config declares any
momentum pair**, so this path is currently inert — available, not dead.

### `resolve_feature_columns()`

The inclusion-mode column list. Must stay consistent with `create_delta_features` or the
"MISSING FEATURE" warning fires. Note the commented-out `indifference_flag` block near the
`home_and_road` special case — `indifference_flag` is handled by the generic
`FEATURE_GROUP_PREFIXES` loop, so the block is a leftover; it also names the wrong columns
(`days_at_home`/`days_on_road`).

### Selection mode

`_defaults.yaml` sets `selection_mode: "inclusion"` — **only columns
`resolve_feature_columns` returns ever reach a model**, regardless of what the ETL built.
This is the mechanism that made barren columns possible and invisible: an extra column in
`games_features.csv` costs nothing visible, it is simply never selected.

`"exclusion"` (legacy) instead drops `metadata_columns + intermediate_columns + target`
and keeps the rest. No shipped config uses it.

---

## 3. Redundancy map

**The rule.** Where one feature group is computed from another, **or** the two measure
Pearson |r| > 0.95 on the training rows, exactly one of them may be enabled in a shippable
config. The comment at the disabled group names the winner and why. Enforced by
`tests/test_config_loading.py::test_no_coupled_pairs_co_enabled`, which reads the
`COUPLED_GROUPS` map in that file.

### Why Boruta-SHAP cannot arbitrate a coupled pair

This is the operating principle behind the rule, and it was learned the expensive way.

Boruta confirms a feature by checking that its mean |SHAP| beats the best shuffled shadow
in significantly more than half of 20 iterations. Two features at r ≈ 1 **split the SHAP
credit between them** — a tree uses whichever it happens to draw at each split, so each
accumulates roughly half the importance the pair would have alone. Half of a large number
still comfortably beats a shadow. So both get confirmed, and the artifact reads exactly like
"two independent signals" while describing one signal counted twice.

> **Boruta confirms that a feature beats noise. It does not confirm that a feature beats its
> own correlate.** For a coupled pair the decision comes from mechanism and held-out
> metrics, never from the selection artifact.

On 2026-08-12 this audit read a `feature_audit` artifact the wrong way, enabled
`point_differential` beside `norm_point_differential` (r = 0.998) and `gds` beside its own
parent `sos_adj_record` (r = 0.946), and test accuracy fell 66.44% → 65.78%. Both were
reverted. See §7.

**The reverting run makes the mechanism visible.** Same data, same splits, same seed — the
only change is that the duplicates are gone. Importances consolidate onto the surviving
member, and features the duplicate had displaced come back:

| feature | with duplicates (40 offered) | decoupled (34 offered) | |
|---|---|---|---|
| `norm_pts_diff_avg_L82_delta` | 0.111 confirmed | **0.280** confirmed | credit it was splitting with raw L82 (0.128) returns — 0.111 + 0.128 ≈ 0.24, close to the consolidated value |
| `norm_pts_diff_avg_L55_delta` | 0.122 | **0.145** | |
| `norm_pts_diff_avg_L34_delta` | 0.038 | **0.094** | its raw twin scored 0.072 |
| `sos_adj_record_L41_at_location_delta` | **rejected** | **confirmed** 0.026 | `gds`, built from it, was taking its place |
| `sos_adj_record_L20_at_location_delta` | tentative 0.022 | **confirmed** 0.030 | same |
| `games_behind_leader_delta` | 0.033 | **0.051** | uncoupled, and still gains from the noise reduction |

A feature whose importance nearly triples when you delete its neighbour was never carrying
independent signal. This table is the argument for the rule, in one place.

### The derivation graph

Every edge below is a place where the ETL computes one group's columns using another's.
`A → B` reads "B is built from A".

```
                      pts_diff  (raw column, aggregator.py)
                       ├──────────────→ point_differential      pts_diff_avg_L*
                       └──────────────→ norm_point_differential norm_pts_diff_avg_L*
                                        (÷ season_avg_total_pts)

  record ──┐
           ├──→ sos_adj_record ──→ gds
  sos    ──┘         (Q)              ↑
    │                                 │
    │                  record (cum win%) ┘   [fallback]
    │
    └──→ adjusted_last_season_record ←── last_season_record

  last_season_record ──[terminal fallback]──→ sos, gds

  eliminated_from_playoffs ┐
  clinched_final_seed      ┴──→ indifference_flag

  rest grid ──→ rested_days, back_to_back, days_at_home, days_on_road

  east_record_* , west_record_* ──→ the two conference features
```

| edge | the function that creates it |
|---|---|
| `record` + `sos` → `sos_adj_record` | `sos_adjusted_record.py::calculate_sos_adjusted_record` — `record_L{lag} * (sos_L{lag}/league_avg)**alpha` |
| `sos_adj_record` → `gds` | `game_difficulty_score.py::_add_opponent_quality` via `_find_largest_sos_adj_col()` → `sos_adj_record_L82` is opponent quality Q |
| `record` → `gds` | `_build_cum_win_pct_lookup` — the first fallback for Q |
| `last_season_record` → `gds`, `sos` | `build_prior_season_strength` — terminal fallback when an opponent has no games yet |
| `last_season_record` + `sos` → `adjusted_last_season_record` | `last_season_record.py::_apply_sos_to_season_record`, using end-of-season `sos_L82` |
| `pts_diff` → both point-differential families | `calculate_pts_diff` and `_compute_norm_pts_diff` are siblings off one column |
| `eliminated_from_playoffs` + `clinched_final_seed` → `indifference_flag` | `playoff_standings.py::compute_playoff_flags`, final line |
| rest grid → 4 columns | `rest_days.py::make_rested_days_table_season` |
| `east_record_*` → conference features | `engineering.py::create_conference_features` |

### Measured correlations

Pearson r on the deltas, 49,175 rows, seasons ≥ 1980/81, after `dropna()`.

| pair | r | over threshold? | resolution |
|---|---|---|---|
| `pts_diff_avg_L*` vs `norm_pts_diff_avg_L*` | **0.998** (0.9967–0.9980, every one of 13 lags) | yes | `norm_point_differential` |
| `last_season_record` vs `adjusted_last_season_record` | **0.9985** (both plain and at-location) | yes | `adjusted_last_season_record` |
| `record` vs `sos_adj_record` | **0.95** (0.88 at L1 → 0.95 at L34+) | yes, at the lags that ship | `sos_adj_record` |
| `gds` vs `sos_adj_record` | 0.946 | under 0.95, but **derivation edge** | `sos_adj_record` |
| `gds_L82` vs `record_L82` | **0.967** | yes | `sos_adj_record` |
| `conf_rank` vs `record_L82` | −0.92 | no | both may ship; `record` is off anyway |
| `games_behind_leader` vs `record_L82` | −0.80 | no | both may ship |
| `rested_days` vs `back_to_back` | −0.53 | no | both enabled; Boruta picks |

### Training-facing member per cluster

| cluster | ships | disabled in its favour | why this one |
|---|---|---|---|
| point differential | **`norm_point_differential`** | `point_differential` | Era-neutral. Training spans 1980/81 to now and scoring pace moved substantially; the raw margin's only extra information *is* the era pace that normalization removes. Also the incumbent. |
| record / schedule strength | **`sos_adj_record`** | `record`, `sos`, `gds` | The adjustment is the entire point: `record` is its unadjusted input, `sos` its multiplier, `gds` its child. Shipping any of them beside it hands the model a product and its own factors. |
| last season | **`adjusted_last_season_record`** | `last_season_record` | Same quantity corrected for the schedule that produced it. The adjusted at-location variant is a top-5 feature in every run; the raw pair is rejected. |
| opponent quality | **`sos_adj_record`** over `gds` | `gds` | Felipe's call. `gds` is the derived member, and `sos_adj_record` must keep being *computed* either way since `gds` cannot be built without it — so retiring the parent would be retiring the child too. |
| rest | **both** `rested_days` and `back_to_back` | — | r = −0.53 is under threshold, so selection is allowed to arbitrate. It does: `back_to_back` is confirmed in every run, `rested_days` rejected in every run. That is redundancy resolving itself correctly, which is what Boruta *is* good for when features are not near-duplicates. |
| playoff standings | **`games_behind_leader`** (one of six exploded groups — see note below) | — | No derivation edge to an enabled group: `compute_playoff_flags` recomputes W/L from `win_bool` itself rather than reading the `record` tables. League-relative where `record` is absolute — two teams on 30-20 sit in different playoff positions depending on conference. `conf_rank` (precursor only, not a group — see `INTERMEDIATE_COLUMNS`) at −0.92 is the closest to the line; re-measure if `record` is ever re-enabled. |

The single `playoff_standings` group was exploded into six independent groups
(`games_behind_leader`, `games_behind_above`, `games_ahead_of_below`,
`clinched_playoff_berth`, `eliminated_from_playoffs`, `clinched_final_seed`) on
2026-08-12 so each can be enabled and tested on its own instead of as a block.
Only `games_behind_leader` ships (`enabled: true` in `_defaults.yaml`); the
other five are `enabled: false` there and `enabled: true` in
`feature_audit.yaml`, which gives `eliminated_from_playoffs`/
`clinched_final_seed` their first-ever Boruta pass — they were previously
unreachable (see §5). `conf_rank` was never one of the six; it moved to
`INTERMEDIATE_COLUMNS` as a documented precursor.

### Configs checked against the map

As of 2026-08-12, after the revert:

| config | enabled groups | coupled pairs co-enabled |
|---|---|---|
| `xgboost.yaml` | norm_point_differential, sos_adj_record, distance, rested_days, back_to_back, adjusted_last_season_record, home_and_road, indifference_flag, games_behind_leader | **none** |
| `all_models.yaml` | identical to xgboost | **none** |
| `llm_features.yaml` | the above + streak, home_and_road raw | **none** |
| `feature_audit.yaml` | everything | **all 7, by design — exempt** |

Two violations existed before this pass and are now fixed:
`point_differential` + `norm_point_differential` and `sos_adj_record` + `gds`, in all three
shippable configs. A third predated the audit entirely: `llm_features.yaml` enabled `record`
while inheriting `sos_adj_record`. That one mattered *more* than the sklearn cases, not less
— nothing trains from that config, so there is no Boruta pass to filter a duplicate out, and
the redundant column goes straight into the prompt spending tokens to restate the next line.

### The `features.yaml` lag-inheritance fragility

A group enabled in `_defaults.yaml` with **no `lags:` override** silently takes whatever
`features.yaml` declares for the ETL. Audited: only `distance` currently does this
(`[1, 3, 7, 14]`), intentionally. `sos_adj_record` is safe because `features.yaml` gives it
only `location_lags`, so `lags` defaults to `[]` and the plain rolling variants never reach
a model.

The hazard is real though: adding a lag to `features.yaml` for an ETL reason changes the
*training* feature set of every enabled group that has no override. Check the enabled-groups
dump after any edit to `features.yaml`.

---

## 4. Column ledger

`data/processed/regular_season/games_features.csv`, 228 columns as of the audit.
Boruta status from `outputs/feature_audit` pass B (every group enabled, matched lags)
unless noted.

### Active — offered to a model

All 34 columns the shipping config offers Boruta. Two status columns, because a feature's
status depends on what it competes against: **shipping** = `outputs/xgboost` (run
`171c718d`, 34 offered, 11 confirmed + 4 tentative = 15 trained on); **audit** =
`outputs/feature_audit` pass B (94 offered, every group on, coupled pairs included — read
§3 before drawing anything from that column).

| Column | Group | shipping | audit | Notes |
|---|---|---|---|---|
| `norm_pts_diff_avg_L82_delta` | `norm_point_differential` | ✅ **0.280** (#1) | ✅ 0.120 | **lag added by this audit.** Importance more than doubled once its raw twin was removed |
| `norm_pts_diff_avg_L55_delta` | `norm_point_differential` | ✅ 0.145 (#2) | ✅ 0.114 | |
| `adjusted_last_season_record_at_location_delta` | `adjusted_last_season_record` | ✅ 0.109 (#3) | ✅ 0.094 | top-5 in every run |
| `norm_pts_diff_avg_L34_delta` | `norm_point_differential` | ✅ 0.094 | ✅ 0.036 | |
| `back_to_back_delta` | `back_to_back` | ✅ 0.085 | ✅ 0.084 | top-6 in every run |
| `games_behind_leader_delta` | `games_behind_leader` | ✅ 0.051 | ✅ 0.029 | **group created by this audit, later exploded from `playoff_standings` into its own group (2026-08-12)** |
| `norm_pts_diff_avg_L41_at_location_delta` | `norm_point_differential` | ✅ 0.042 | ✅ 0.040 | |
| `norm_pts_diff_avg_L13_delta` | `norm_point_differential` | ✅ 0.042 | ✅ 0.020 | |
| `sos_adj_record_L20_at_location_delta` | `sos_adj_record` | ✅ 0.030 | ✗ | confirmed → tentative → confirmed as `gds` was enabled then reverted |
| `sos_adj_record_L41_at_location_delta` | `sos_adj_record` | ✅ 0.026 | ✗ | confirmed → **rejected** → confirmed, same cause |
| `norm_pts_diff_avg_L8_delta` | `norm_point_differential` | ✅ 0.026 | ~ 0.017 | |
| `norm_pts_diff_avg_L21_delta` | `norm_point_differential` | ~ 0.023 | ✗ | |
| `indifference_flag_delta` | `indifference_flag` | ~ 0.021 | ~ 0.019 | sign is inverted, see §2 |
| `distance_L14_delta` | `distance` | ~ 0.021 | ✗ | |
| `norm_pts_diff_avg_L20_at_location_delta` | `norm_point_differential` | ~ 0.020 | ✗ | |
| `days_at_home_delta` (= `days_at_home + days_on_road`) | `home_and_road` | ✗ 0.018 | ✗ | raw pair kept instead in `llm_features.yaml` |
| `adjusted_last_season_record_delta` | `adjusted_last_season_record` | ✗ 0.018 | ✗ | the at-location variant carries the group |
| `norm_pts_diff_avg_L{1,3,5}_delta`, `..._L{5,10}_at_location_delta` | `norm_point_differential` | ✗ | ✗ | short windows, offered and dropped every run |
| `sos_adj_record_L{5,10}_at_location_delta` | `sos_adj_record` | ✗ | ✗ | |
| `distance_L{1,3,7}_delta` | `distance` | ✗ | ✗ | |
| `rested_days_delta` | `rested_days` | ✗ 0.005 | ✗ | rejected in every run; `back_to_back` carries the rest signal |
| `games_behind_above_delta`, `games_ahead_of_below_delta`, `clinched_playoff_berth_delta` | `games_behind_above`, `games_ahead_of_below`, `clinched_playoff_berth` | ✗ | ✗ | offered each run, dropped each run. `conf_rank` never produces a `_delta` column — it is a precursor in `INTERMEDIATE_COLUMNS`, not a group |
| `eliminated_from_playoffs_delta`, `clinched_final_seed_delta` | `eliminated_from_playoffs`, `clinched_final_seed` | — | — | reachable for the first time as of the 2026-08-12 group split (previously computed but unreachable by any group — see §5); not yet run through Boruta |
| `conference_diff_home_advantage_pct`, `conference_home_court_advantage_pct` | built in `engineering.py` | ✗ | ✗ | rejected in **every** run examined — see §6 |
| `streak_delta`, `days_at_home`, `days_on_road` | various | — | — | `llm_features.yaml` only; **never measured**, nothing trains from that config |

✅ confirmed ~ tentative ✗ rejected — importances are mean |SHAP| averaged over 20 iterations

### Computed but deliberately not offered — the coupled members

These are **not barren**: they are live groups with a documented reason to stay off, and
`_defaults.yaml` records the correlation and the experiment to run if anyone revisits them.

| Column family | Group | Why off | Last measured |
|---|---|---|---|
| `pts_diff_avg_L*_{HT,VT}`, `..._at_{home,road}` | `point_differential` | r = 0.998 with `norm_point_differential` | ✅ L82 0.129, L34 0.067, L13 0.025 (audit pool) |
| `gds_L*_{HT,VT}`, `..._at_{home,road}` | `gds` | computed from `sos_adj_record`; r = 0.946 | ✅ L82 0.083, L55 0.023 (audit pool) |
| `record_L*` | `record` | input to `sos_adj_record`; r = 0.95 | ✗ all 13, best 0.008 |
| `sos_L*` | `sos` | input to `sos_adj_record` | ✗ all 18, best 0.015 |
| `last_season_record_*` | `last_season_record` | r = 0.9985 with the adjusted version | ✗ both, best 0.014 |
| `streak_{HT,VT}` | `streak` | measured rejection, not coupling | ✗ 0.007 |
| `neutral_court` | `neutral_court` | 0.68% of games; importance exactly 0.000 | ✗ 0.000 |

### Precursor — consumed to build something else

| Column | Feeds | Where consumed |
|---|---|---|
| `east_record_adjusted`, `west_record_adjusted`, `east_record_at_east`, `west_record_at_west` | the two conference features | `create_conference_features`, then dropped |
| `record_L82_HT`, `record_L82_VT` | `record_L82_delta` for `RecordDifferenceBaseline` | `splits.py` builds it onto `X_test_baseline` |
| `pts_diff_avg_L82_HT`, `pts_diff_avg_L82_VT` | `pts_diff_avg_L82_delta` for `PointDifferentialBaseline` **and** the active `pts_diff_avg_L82_delta` feature | `splits.py`; baseline re-enabled by this audit |
| `eliminated_from_playoffs_{HT,VT}`, `clinched_final_seed_{HT,VT}` | `indifference_flag` (still) **and**, as of the 2026-08-12 group split, their own `eliminated_from_playoffs`/`clinched_final_seed` groups | `indifference_flag` is computed inside `compute_playoff_flags`, **before** the merge — that dependency is unaffected. The merged `_HT`/`_VT` copies are no longer barren: they now also feed their own delta columns (see §4) |
| `conf_rank_{HT,VT}` | `games_behind_leader`, `games_behind_above`, `games_ahead_of_below`, and the clinch logic | computed inside `compute_playoff_flags`; listed in `INTERMEDIATE_COLUMNS`, deliberately never its own group |
| `total_wins_*`, `total_losses_*`, `games_played_HT_at_home`, `games_played_VT_on_road`, `pts_diff`, `distance` | divisors / intermediates | listed in `INTERMEDIATE_COLUMNS`; `pts_diff` also feeds every point-differential builder |
| `season_avg_total_pts` | `norm_pts_diff` denominator | internal to `point_differential.py`, never merged |
| `is_neutral_court_game` | `neutral_court` fallback when `arenaId` is null | `add_neutral_court`; also metadata for plotting |
| `winnerteamConference` | East/West win attribution | `make_east_west_record`; also metadata |

### Metadata — not model inputs, by design

`gameId`, `gameDate`, `gameDateOnlyStr`, `season`, `hometeamPrename`, `hometeamName`,
`hometeamId`, `awayteamPrename`, `awayteamName`, `awayteamId`, `homeScore`, `awayScore`,
`winner`, `overtimes`, `postponed`, `gameType`, `hometeamLocation`, `gameLocation`,
`awayteamLocation`, `hometeamConference`, `awayteamConference`, `winnerteamConference`,
`is_neutral_court_game`, `win_bool` (target), `games_played_HT`, `games_played_VT`.

`games_played_HT` / `games_played_VT` are not plotting columns: `weighting.py` ramps each
row's sample weight over the first `saturation_K` games of a season using
`min(games_played_HT, games_played_VT)`, and `splits.py` uses them for the
`minimum_games_train` / `minimum_games_test` masks (the masks read them off
`games_enriched`, the untouched frame, so they work either way — the weighting does not).

`_defaults.yaml` listed both; `DEFAULT_METADATA_COLUMNS` in `src/config/constants.py` did
not, so a config that declined to override `metadata_columns` would have taken a `KeyError`
in `compute_sample_weights`. Every shipped config includes `_defaults.yaml`, so nothing hit
it. The two lists were brought into agreement in this audit — they are now identical.

`overtimes`, `postponed`, `gameType`, `is_neutral_court_game` drive the per-segment accuracy
tables in `src/ml/evaluation/analysis.py`.

### Barren — computed, reachable by nothing

See §5. All were found in this audit; each row states what was done.

---

## 5. Barren columns found, and what was done

One row here is genuinely barren — the playoff standings columns, which had no config knob
and could not reach a model by any path. The rest turned out to be **precursors or
deliberately-disabled coupled members** once the redundancy map (§3) existed: a column that
exists because `sos_adj_record` needs it, or because it is the raw twin of the feature that
ships, is doing a job. The audit's first pass called several of these barren and re-enabled
two; that was the error §3 now prevents.

| Columns | Count | Status when found | Action |
|---|---|---|---|
| `conf_rank_{HT,VT}`, `games_behind_leader_{HT,VT}`, `games_behind_above_{HT,VT}`, `games_ahead_of_below_{HT,VT}`, `clinched_playoff_berth_{HT,VT}` | 10 | **No feature group existed.** `compute_playoff_flags` emitted them, `aggregator.py` merged them, and `resolve_feature_columns` had no entry for them, so inclusion mode dropped all ten every run. `INTERMEDIATE_COLUMNS` still carries `# "conf_rank_HT", # "conf_rank_VT"` commented out — the trace of an unfinished attempt. | **Fixed.** Added a `playoff_standings` group (`FEATURE_GROUP_PREFIXES`, `FeaturesMapConfig`, `_defaults.yaml`). `games_behind_leader_delta` is now confirmed at 20/20 hits in both audit passes. **Further exploded 2026-08-12 — see next row.** |
| `games_behind_above_{HT,VT}`, `games_ahead_of_below_{HT,VT}`, `clinched_playoff_berth_{HT,VT}` (0/20 hits inside the bundle above), `eliminated_from_playoffs_{HT,VT}`, `clinched_final_seed_{HT,VT}` (never reachable at all) | 10 | The bundled `playoff_standings` group forced these five to share `enabled`/`delta` with the confirmed `games_behind_leader`, and gave the two precursor columns no path to a model. `conf_rank_{HT,VT}` stayed uncommented-out and unreachable. | **Fixed 2026-08-12.** Exploded into six independent groups: `games_behind_leader`, `games_behind_above`, `games_ahead_of_below`, `clinched_playoff_berth`, `eliminated_from_playoffs`, `clinched_final_seed` (`FEATURE_GROUP_PREFIXES`, `FeaturesMapConfig`, `_defaults.yaml`, `feature_audit.yaml`). `eliminated_from_playoffs`/`clinched_final_seed` get their first Boruta pass via `feature_audit.yaml`. `conf_rank` moved into `INTERMEDIATE_COLUMNS` (uncommented) as a documented precursor, not a group. |
| `sos_L{1..82}_{HT,VT}` | 18 | Same shape: `sos_adj_record` and `gds` consume the in-memory SOS table upstream, never these merged columns. Group was `enabled: false`. | **Measured, left off.** All 18 rejected (best 0.015 at 5/20). `_defaults.yaml` now records that rather than leaving a bare `enabled: false`. |
| `sos_adj_record_L{1..82}_{HT,VT}` (non-location) | 18 | The plain rolling variants; `_defaults.yaml` declares `location_lags` only. Also feed GDS upstream, in-memory. | **Left off, comment already existed** ("the plain rolling variants lost to the at-location ones"). |
| `record_L*_{HT,VT}`, `record_L*_at_{home,road}` | 26 | `record` is off everywhere: it is the unadjusted input to `sos_adj_record`, r = 0.95. | **Left as-is, now with the coupling documented.** The ETL must keep building them — `calculate_sos_adjusted_record` intersects `record_lags` with `sos_lags`, so trimming the list breaks the feature that replaced them. **Precursor, not barren.** |
| `pts_diff_avg_L*_{HT,VT}`, `..._at_{home,road}` | 26 | Group disabled: r = 0.998 with `norm_point_differential`. | **Briefly re-enabled by this audit, then reverted.** Confirmed by Boruta (L82 0.129, L34 0.067) but that was the coupled-pair artifact — see §3. Two columns are genuine precursors: `pts_diff_avg_L82_{HT,VT}` feed `PointDifferentialBaseline`. |
| `gds_*` | 24 | Group `enabled: false`, no comment, **never once through feature selection**. | **Measured, then reverted for coupling.** Confirmed at L82 (0.083–0.097, 5th of 40) and L55, but computed from `sos_adj_record` at r = 0.946, and it displaced its own parent when co-enabled. `_defaults.yaml` now records the measurement, the coupling and the experiment to run if revisited. |
| `neutral_court` | 1 | Group `enabled: false`, no comment, never measured. | **Measured, left off** with the result recorded: importance exactly 0.00000, 0 hits, both passes. |
| `last_season_record_*` | 4 | Superseded by the adjusted version, r = 0.9985. | **Measured, left off**, coupling now recorded at the point of disablement. **Precursor** — `_prior_season_lookup` output is the input the adjusted builder corrects. |
| `streak_{HT,VT}` | 2 | Off for sklearn (measured rejection, 0.007 — not a coupling case), on for the LLM config. | **Left as-is.** The one group that is off for the plain reason of having no signal. |
| `pts_diff_avg_L82_delta` (derived) | 1 | `splits.py` built it onto `X_test_baseline` every run and `PointDifferentialBaseline` was **commented out** in `experiment.py`, behind a stale `TODO: make this feature available even when it's not passed to train config` — a blocker `splits.py` had already solved. | **Fixed.** Baseline re-enabled, mirroring the record baseline block. |

**Not barren, but flagged:** `momentum_pairs` machinery in `engineering.py` and
`schema.py` is complete and correct but no config declares a pair. Inert rather than dead —
left alone.

---

## 6. Open questions for Felipe — design calls I did not make

1. **The two conference features are rejected by Boruta in every run.**
   `conference_diff_home_advantage_pct` and `conference_home_court_advantage_pct` were the
   stated reason one model can serve every game after the conference split was removed
   ("the conference signal stayed. Two features carry it"). Feature selection drops both,
   in the shipping pool and in both audit passes, at importance ~0.006. So the deployed
   model contains **no conference signal at all**. The split's removal is still backed by
   its own eight-season backtest — this does not argue for bringing it back — but the
   comment in `xgboost.yaml` describes something that is not happening. Either the features
   need rework (they are league-level, so they vary only by date, which is very little
   signal per row), or the comment should be corrected. **I changed neither.**

2. **`prepare_data` calls `df.dropna()` on the whole frame**, before feature selection.
   That costs **570 training rows** whose every feature is present, purely because
   `gameType` — a *metadata* column — is NaN on them (608 NaN rows total, post-1980/81).
   Restricting the `dropna` to feature + target columns would recover them, but it changes
   the row set for every model and every historical comparison. That is your call, not
   mine.

3. **`llm_features.yaml` never gets a Boruta pass.** Nothing trains from it, so its `streak`
   and raw `days_at_home`/`days_on_road` columns reach the prompt unfiltered. That is
   arguably correct — the argument for them is semantic legibility, not SHAP importance —
   but it means the LLM's feature set is the one part of the pipeline chosen entirely by
   hand, and therefore the one place the redundancy rule has to be enforced by reading
   rather than by measurement. `record` was removed from it on exactly that basis.

4. **Promotion.** The decoupled run (`171c718d`) beats the deployed model on all four test
   metrics — see §7. `configs/predict/predict_classifier.yaml` still points at `ee34b43e`,
   because training does not deploy in this project. Promoting is
   `make train TRAIN_CONFIG=xgboost PROMOTE=1`.

### Resolved by this pass

- ~~`gds` displaced `sos_adj_record`~~ — **resolved, and it became the worked example.**
  Enabling `gds` beside its own parent pushed `sos_adj_record_L41_at_location` from
  confirmed to rejected and L20 from confirmed to tentative; reverting `gds` restored both
  to confirmed. That is substitution between coupled features in both directions, observed
  twice. The pair is resolved in favour of the parent (§3), and `sos_adj_record` keeps being
  *computed* regardless, since `gds` cannot be built without it.

---

## 7. What the 2026-08-12 audit actually bought

Three runs, same data, same splits, same seed. The middle column is the mistake; the right
column is the same audit findings with the coupled pairs removed.

| | deployed `ee34b43e` | coupled (§3 violated) | **decoupled `171c718d`** |
|---|---|---|---|
| features offered to Boruta | 28 | 40 | 34 |
| features trained on | 14 | 18 | 15 |
| test accuracy | 66.44% | 65.78% | **67.19%** |
| test Brier | 0.2094 | 0.2100 | **0.2092** |
| test ROC AUC | 0.7257 | 0.7267 | **0.7271** |
| test ECE | 0.0350 | 0.0428 | **0.0268** |

The decoupled run is best on all four. Read the margins honestly:

- **Accuracy +0.75pp** over deployed — 9 games in a 1,201-row test set. Real but not large.
- **ECE 0.0350 → 0.0268**, a 23% reduction in calibration error. The most substantive gain,
  and the one that matters most downstream: bet selection consumes probabilities, not
  labels.
- **Brier −0.0002** — inside noise on its own. This project measures SEs of 0.00065–0.00096
  on this split (see the `weighting:` comment in `_defaults.yaml`).
- **AUC +0.0014** — small, same direction.

The like-for-like comparison that isolates the redundancy is the middle column against the
right: **+1.41pp accuracy and 37% less calibration error, from deleting 6 columns that
duplicated 6 others.** That is Felipe's hypothesis, confirmed.

What survived the revert, and why it is not nothing:

- `norm_pts_diff_avg_L82_delta` — previously excluded by an undocumented lag cap. Now the
  strongest feature in the model at 0.280, more than double the next.
- the `playoff_standings` group — 10 columns that had no config knob at all and could not
  reach a model by any path. `games_behind_leader_delta` confirmed at 20/20 hits in four
  independent runs. Later exploded (2026-08-12) into six independent groups —
  `games_behind_leader`, `games_behind_above`, `games_ahead_of_below`,
  `clinched_playoff_berth`, `eliminated_from_playoffs`, `clinched_final_seed` — so each can
  be enabled and tested on its own; only `games_behind_leader` ships.
- `gds` and `point_differential` were reverted, but they are no longer *unmeasured*. Their
  `enabled: false` now carries the correlation, the Boruta result, and the experiment to run
  if anyone revisits them.

**Not promoted.** `configs/predict/predict_classifier.yaml` still points at `ee34b43e`.
This is a promotion candidate on the numbers — `make train TRAIN_CONFIG=xgboost PROMOTE=1`
— but training does not deploy in this project, and that call is Felipe's.

### The baselines

`PointDifferentialBaseline` was re-enabled in this audit (it had been commented out behind a
stale TODO). Two reference lines now print on every run:

| model | test accuracy | test ROC AUC |
|---|---|---|
| record baseline (`record_L82_delta` threshold) | 65.28% | 65.21% |
| point differential baseline (`pts_diff_avg_L82_delta` threshold) | 65.45% | 65.43% |
| decoupled xgboost | **67.19%** | **72.71%** |

A single-threshold rule on one column is within 1.74pp of the tuned model on accuracy. The
model earns its keep on **ranking** — 7.3 points of AUC — which is the quantity bet
selection needs. Keep this table in view whenever an accuracy delta of half a point is being
treated as signal.

Note both baselines read raw `pts_diff_avg_L82` / `record_L82` columns, from groups that are
`enabled: false`. That is not a redundancy-rule violation: `splits.py` builds
`X_test_baseline` from the untouched feature frame *before* `create_delta_features` runs, so
the baselines are independent of group enablement by construction.

---

## 8. Proposing new features

Think about causal factors that exist *before* tip-off:
- Head-to-head record within the current season (the H2H table already exists inside
  `playoff_standings.py`, unexported)
- Season stage interacted with standings position
- Roster availability / injuries (not currently ingested)
- Days since the last game against *this specific* opponent

For each proposal, specify:
1. The lag/window (if applicable)
2. Which teams (HT/VT or both, or location-scoped)
3. Whether a delta makes sense — and if so, difference or sum (see `home_and_road`)
4. The neutral value when the feature is undefined off-population (see
   `create_conference_features` — a feature defined on a sub-population gets imputed at its
   neutral value, never dropped and never NaN)
5. Confirm no post-game data is used, at **day** granularity, not timestamp
6. The join pattern (see SKILL.md Section 6)
7. **What it correlates with.** Before enabling anything, run the check below against every
   already-enabled group. If it is built from one of them, or lands over 0.95, it does not
   get enabled alongside — one of the two goes, and the loser's `enabled: false` gets a
   comment saying which won and why. Add the edge to `COUPLED_GROUPS` in
   `tests/test_config_loading.py`.

```python
# The check that would have prevented the 2026-08-12 regression.
import pandas as pd, numpy as np
df = pd.read_csv("data/processed/regular_season/games_features.csv")
df = df[df.season >= "1980/81"].dropna()
d = lambda b: df[f"{b}_HT"] - df[f"{b}_VT"]
print(np.corrcoef(d("my_new_feature_L34"), d("norm_pts_diff_avg_L34"))[0, 1])
```

And then: add it to `FEATURE_GROUP_PREFIXES` and `FeaturesMapConfig` as well as the YAML.
A merged column with no feature group is invisible — that is how ten of them accumulated.

**Do not reach for "enable it and let Boruta decide".** That is safe only for a feature
uncorrelated with what is already there. For anything derived from an existing feature it is
actively misleading: both members get confirmed, the artifact looks like two signals, and
the model gets worse. §3 has the mechanism and the measured example.
