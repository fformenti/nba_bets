# Pipeline audit — 2026-08-11

Written after a full cold-start run of the pipeline from raw historical data
only: parse → train → fetch slate → predict → fetch outcomes → score → append.
Every finding below was read out of the code and, where marked **verified**,
reproduced during that run.

Findings are ordered by how much damage they can do, not by how hard they are to
fix.

**Status: F1–F6 are fixed.** Two findings were corrected after the first draft,
once measured rather than reasoned about — F5 was narrower than it looked and
F6 was simply wrong about where the time goes. Both corrections are recorded
below rather than quietly edited away.

---

## F1 — Last-season features are always NaN at prediction time (high, verified)

`run_prediction_pipeline` narrows history to the seasons in the slate before
building features:

```python
# src/ml/prediction/pipeline.py:266
historical_features = historical_features[
    historical_features["season"].isin(upcoming_seasons)
].copy()
```

`_prior_season_lookup` (`src/etl/features/last_season_record.py:79`) then has no
prior-season rows to join against, so every team's previous-season record
resolves to NaN — and the expansion-team fallback (`prev_season_min`) is NaN for
the same reason.

Verified in this run's prediction log, for all three models:

```
Rows with missing values: 6 out of 6
Features with missing values:
{'adjusted_last_season_record_delta': 6,
 'adjusted_last_season_record_at_location_delta': 6}
```

This is not a cosmetic gap. `adjusted_last_season_record_delta` and
`adjusted_last_season_record_at_location_delta` were both **selected by
Boruta-SHAP** for the same-conference model (2 of its 16 features), so the model
was trained on a real signal it can never see in production. `allow_missing_features: true`
in `configs/predict/predict_classifier.yaml` turns what would be a loud
`ValueError` into an INFO line, and the imputer fills the hole with a column
mean. Nothing downstream fails; the model just gets quietly worse.

**Fix.** Keep the prior season in the frame:

```python
seasons = set(upcoming_seasons) | {_prev_season(s) for s in upcoming_seasons}
historical_features = historical_features[historical_features["season"].isin(seasons)]
```

**Fixed.** The filter now keeps the slate's seasons *and* their predecessors, and
`_prev_season` was promoted to a public `prev_season` since the prediction
pipeline is a legitimate caller.

**The contract is hardened too.** `_align_features` now separates "NaN for some
rows" (ordinary — a team with no games played yet has no rolling average) from
"NaN for every row of the slate" (a train/serve skew: the model was fitted on a
real signal and is being handed an imputed constant). The second warns under
`allow_missing_features` and raises without it.

Verified on the next slate (2026-01-26, 7 games): all three models now report
`No missing values found in aligned features`, where every row previously carried
two empty features.

Live accuracy over the two slates scored so far moved from 33.3% (12 rows,
pre-fix) to 61.5% cumulative (26 rows), with Brier 0.356 → 0.246 and ECE
0.399 → 0.120. The post-fix slate alone went 12/14. **Two slates prove nothing
causally** — the first was a road-heavy night — but the direction is at least not
contradicting the fix.

---

## F2 — `full-rebuild` cannot actually rebuild (high)

Two reference tables are load-bearing and unreproducible:

- `data/processed/TeamsHistoriesLocationsNBALookUpTable.csv` — read by
  `get_teams_locations` (`src/etl/utils/common.py:119`), which
  `parse_raw_games` calls. Without it, **step one of the pipeline fails.**
- `data/processed/locations_distances.csv` — read by every travel feature
  (`src/etl/features/distances.py:93`).

Their only builders call paid APIs (`teams_locations.py` → OpenAI,
`distances.py` → Serper + OpenAI), `data/` is gitignored, and the `full-rebuild`
target calls `build-distances-table` while `build-teams-locations` — which
produces its input — is commented out two lines above. So `make full-rebuild`
on a clean checkout fails, and the fallback path costs money.

Worse, `build_distances_table` still carries a debug slice:

```python
# src/etl/reference/distances.py:100
for pair in tqdm(all_combinations[0:5]):
```

Running it would overwrite a good 561-row distance table with 5 rows. Every
`distance_L*` feature would go NaN, and with `drop_na: true` that silently
shrinks the training set rather than erroring.

This run recovered both files by copying them from an old checkout at
`../old/nba_bets/data/processed/`. That is not a recovery strategy.

**Fixed:**

1. Both lookups moved to `data/raw/reference/`. They are inputs, not build
   artifacts, and they now sit with the other inputs. `paths.py` gained
   `RAW_REFERENCE_DIR`, and the misleadingly-named
   `TEAMS_CITIES_LOCATIONS_HISTORY_PROCESSED_PATH` became
   `TEAMS_LOCATIONS_REFERENCE_PATH` — it was never "processed".
2. `build-distances-table` is out of `full-rebuild`, which is now offline and
   free by design. The Makefile says why, so it does not creep back in.
3. The `[0:5]` slice is gone, replaced by an explicit `--limit` that warns loudly
   that it is producing a partial table.
4. `require_reference_file` fails early and names the file and the target that
   rebuilds it, instead of a `FileNotFoundError` from three frames inside pandas.
5. Four dangling constants (`RAW_DISTANCES_PATH`, `ALL_TEAMS_HISTORY_PATH`,
   `NBA_TEAMS_HISTORY_PATH`, `TEAMS_HISTORIES_CONFERENCE_NBA_CSV_PATH`) pointed
   at files that do not exist and that nothing reads. Removed — they implied a
   provenance the repo does not have.

**Still open, and it needs a decision from you.** `data/` is gitignored in full,
so moving the two files to `data/raw/reference/` organises them correctly but
does *not* make a fresh clone reproducible. See "What you need to do" below.

---

## F3 — Operator-precedence bug in the postponed mask (medium, verified)

```python
# src/etl/ingestion/raw_games.py:64
mask_postponed = (df["homeScore"] <= 0) | (
    df["awayScore"] <= 0 | df["homeScore"].isna() | df["awayScore"].isna()
)
```

`|` binds tighter than `<=`, so the second clause evaluates as
`awayScore <= (0 | isna | isna)` — a score compared against a boolean. Verified:

| home | away | as written | as intended |
|-----:|-----:|:----------:|:-----------:|
| 100.0 | NaN | False | **True** |
| NaN | 105.0 | False | **True** |
| 0.0 | 98.0 | True | True |
| 110.0 | 99.0 | False | False |

A row with a missing score is not flagged postponed, so it reaches
`process_ingested_games`, passes the `postponed == 0` filter, and lands in the
regular-season table with no outcome.

The current archive happens to have no missing-score rows, so nothing is broken
today. But this function parses every future refresh of `Games.csv`, and a
missing score is exactly what an in-progress or suspended game looks like.

**Fix.** Add the parentheses:

```python
mask_postponed = (
    (df["homeScore"] <= 0) | (df["awayScore"] <= 0)
    | df["homeScore"].isna() | df["awayScore"].isna()
)
```

---

## F4 — Training rewrites a version-controlled config as a side effect (medium)

`_update_predict_config` (`src/ml/training/classifier.py:95`) regex-rewrites
`configs/predict/predict_classifier.yaml` at the end of every training run:

```python
content = re.sub(rf'^(\s+{conference_filter}:\s+").*(")', ...)
```

Two problems. First, **any** training run repoints production inference —
including an experiment, a smoke test, or a run that scored worse than what was
already deployed. There is no gate. Second, the pattern matches any mapping key
named `same`, `different` or `all` at any indentation; it is correct only
because that file currently has exactly one such block.

It is genuinely convenient (it saved a manual step in this run), so the fix is
to keep the automation and lose the hazard.

**Fixed, by making deployment a decision rather than a side effect.** Training no
longer touches the predict config unless asked:

```
make train-all              # trains, logs each run URI, deploys nothing
make train-all PROMOTE=1    # trains and points prediction at the new runs
```

The rewrite itself is also anchored to the `model_uris:` block now, so it cannot
wander into some future key that happens to be named `all:`, and it warns instead
of silently no-op'ing when the entry is absent.

**Not done: the registry-alias version.** `models:/nba_classification_xgboost_same@champion`
is still the better end state — it removes YAML rewriting entirely and makes
"which model is live" a question you ask MLflow rather than git. That is a larger
change to how deployment works and belongs in its own commit, not an audit fix.
The `--promote` gate closes the actual hazard in the meantime.

---

## F5 — Silent zeros in the distance join (medium) — *corrected, then fixed*

**The first draft of this finding was wrong in an important way.** It called the
`fillna(0)` on `driving_distance` a bug outright and proposed leaving unmatched
pairs NaN. That would have broken the feature: `locations_distances.csv` is built
from `itertools.combinations(unique_locations, 2)`, so it holds **distinct pairs
only**. A team that stays in the same city has no row by construction, and the
`fillna(0)` is what correctly records "no travel". Blanket-NaN would have thrown
away every rest day at home.

The real problem is that one `fillna` was covering three different situations:

| situation | correct value | before |
|---|---|---|
| same city (no row by construction) | 0 — genuinely no travel | 0 ✓ |
| different cities, pair missing from the table | unknown — a gap | **0 ✗** |
| location itself unknown for that team-season | unknown | **0 ✗** |

Only the first is legitimately zero. The other two were reading as "the team did
not travel", which is the same value as a rest day at home.

**Fixed** by splitting the three cases: same-city stays 0, a genuinely unmatched
pair between two *known* different cities now raises with the offending pairs
listed, and an unknown location is counted and warned about (it cannot be judged
either way, so it stays 0 rather than failing the build).

Verified against the current data: **0 unmatched pairs across 87,876 real
city-to-city transitions** — the reference table has complete coverage, so the
new raise is dormant on today's data and only fires on a genuine gap. The
unknown-location warning does fire, for 2–36 team-days per season in the 1960s;
those seasons are far below the models' `min_season` of 1980/81.

**Related, not a bug:** `distance_L{lag}` rolling means include the current row,
unlike every other feature, which shifts by one. That is defensible — the
schedule is known before tip-off, so travel into a game is not lookahead — but it
is the one exception to the codebase's otherwise consistent `shift(1)`
discipline. Now documented as such in the code.

---

## F6 — Where `build-features` actually spends its time (low, perf) — *corrected*

**The first draft named the wrong culprit.** It asserted that the row-wise
`.apply(axis=1)` in `rest_days.py` dominated the ~2m30s build. It does not.
Vectorising it with `np.select` produced **byte-identical output across all
278,276 rows** — and moved total runtime from 2:32 to 2:27, i.e. nothing. The
change is kept because it is strictly better and now verified, but it was not
the bottleneck. The claim was reasoning, not measurement.

Profiling `create_features_tables_from_config` over the real 66k-game table:

| function | cumulative | share |
|---|---:|---|
| `playoff_standings.py::compute_playoff_flags` | **206 s** | ~80% |
| `strength_of_schedule.py::_compute_sos_columns` | 60 s | ~24% |
| everything else | ~15 s | |

The cost is Python-level iteration: 27M `numpy.searchsorted` calls and 980k
`fast_xs` (row-at-a-time) accesses, from the `for i in range(n)` loops in
`_compute_sos_columns` and the per-(season, conference, date) loops in
`playoff_standings.py:335-379`.

**This is not dead work.** The obvious move — gate the playoff table off, since
`configs/features.yaml` never lists it — is wrong: `indifference_flag` comes from
that table, and `indifference_flag_delta` was *selected by Boruta-SHAP* for the
same-conference model. It is load-bearing.

**Left unfixed, deliberately.** Both hot spots implement leakage-safe standings
and strength-of-schedule semantics inside those loops. Rewriting them is a real
piece of work on a feature the models consume, not a mechanical vectorisation,
and it deserves its own change with its own equivalence tests rather than being
folded into an audit fix. Two and a half minutes for a full historical rebuild is
tolerable today; this matters when it stops being.

---

## F7 — Prediction rebuilds every feature table from scratch per slate (low)

`build_prediction_feature_base` runs the full `create_features_tables_from_config`
over one season plus the slate, then `merge_features` performs ~24 joins — to
predict 6 games. It is correct and it is fast enough today (~10s), but the work
scales with season length, not slate size, and it runs on every daily cycle.

Not worth changing now. Worth knowing before anyone tries to predict
intraday or backfill a season of slates one day at a time.

---

## F8 — `fillna(0.0)` on rolling averages conflates "no data" with "average" (low)

`pts_diff_avg_L{lag}` and `norm_pts_diff` fill missing rolling values with 0.0
(`src/etl/features/point_differential.py:60,128`). For a normalized point
differential, 0.0 means "exactly league-average team", which is a real claim
about a team with no games played. `filters.minimum_games_train: 5` masks most
of it. Leaving these NaN and letting the configured imputer act would be more
honest, and would let the imputation strategy be tuned in one place.

---

## What is solid

Worth stating plainly, because the fragility above is concentrated in the
reference-data and serving layers, not the modelling core:

- **Leakage discipline is consistent and deliberate.** Every rolling feature
  shifts by one within `(teamId, season)`, the season-normalizing denominator is
  an expanding mean over prior games only, and the comments explain why. F5's
  exception is defensible on its merits.
- **One split definition.** `build_splits` is genuinely the only place splits are
  decided, and `test_llm_split_parity.py` holds that line.
- **The incremental state machine is well designed.** pending → results →
  archive, with postponed and unresolved as explicit parked states, means "where
  is this file?" answers "what happened to this game?". The quarantine grace
  period is what stops one unanswerable game deadlocking the frontier.
- **The history table is defended.** Atomic writes, duplicate-gameId assertions
  before every write, and the documented rule that a postponed archive row never
  supersedes a played one.
- **Prediction upsert semantics are correct** — re-running a slate replaces rows
  rather than appending, so the scorecard cannot double-count.

310 tests pass against the rebuilt data.

---

## Data provenance

Every file under `data/` traces to a writer except two. The full inventory, as of
this audit:

| origin | files |
|---|---|
| **External download** | `raw/historical/games/Games.csv`, `raw/historical/LeagueSchedule25_26.csv` (re-fetchable via `make fetch-league-schedule`) |
| **Handmade** | `raw/historical/handmade/TeamsHistoriesConferenceNBA.csv`, `raw/historical/handmade/polymarket_teams_abv.csv` |
| **Generated by an external service — cannot be rebuilt offline** | `raw/reference/TeamsHistoriesLocationsNBALookUpTable.csv`, `raw/reference/locations_distances.csv` |
| **Pipeline output** | everything else: `ingested/`, `processed/`, `predictions/`, `raw/incremental/` |

The two reference files are the only ones whose contents cannot be re-derived
from anything in this repo. They now live under `raw/` with the other inputs,
which is what they are. Together with the two handmade files they are tracked in
git — see the next section.

## Cold start — resolved by versioning the inputs

**The unreproducible inputs are now tracked in git.** `.gitignore` no longer
ignores `data/` wholesale; it ignores everything under it *except* the files
nothing can regenerate:

```gitignore
/data/*
!/data/raw/
/data/raw/*
!/data/raw/reference/          # LLM/Serper-generated lookups
!/data/raw/historical/
/data/raw/historical/*
!/data/raw/historical/handmade/  # typed by hand
```

Four files, 2,240 lines, ~170KB, changing only when a franchise relocates or a
new team enters. The handmade pair is included for the same reason as the
reference pair: `make build-teams-history` — step one of `full-rebuild` — reads
`TeamsHistoriesConferenceNBA.csv`, and no command anywhere can reproduce it.

A fresh clone now rebuilds with no credentials, no backup and no API spend —
**once `Games.csv` is dropped in**. That is the one remaining gap, and it is
different in kind: `data/raw/historical/games/Games.csv` is a 10MB public
dataset, deliberately left untracked, but *nothing in this repo records where it
came from* and there is no `make` target that fetches it. Restoring it is a
manual step someone has to already know how to do.

The cheap fix is a provenance line, not a 10MB blob in git: add the source URL to
the `ingest-raw-games` docstring so the knowledge outlives whoever has it. I have
not written it because I do not know the source. Everything else under `data/`
stays ignored, as it is all derived.

**Do not let this drift.** The rule these files live under is that
`data/raw/reference/` holds inputs, not artifacts. If a future change starts
writing build output into either directory, it lands in version control by
default — which is exactly backwards.

The two alternatives, recorded in case the tradeoff is ever revisited: keep
`data/` fully ignored and back the files up outside the repo
(`require_reference_file` names what is missing), or rebuild them from the APIs —
now actually possible since the `[0:5]` slice is gone — with
`OPENAI_API_KEY` + `SERPER_API_KEY` and
`make build-teams-locations && make build-distances-table` (~200 + ~780 calls).
If you ever do run that rebuild, diff it against the tracked files rather than
trusting it blind; the git history is now the reference to diff against.

## Still open

- **F6's real hot spots** — `compute_playoff_flags` (206s) and
  `_compute_sos_columns` (60s). Deliberately deferred; see above.
- **F7 / F8** — unchanged, both low. F8 in particular (`fillna(0.0)` on rolling
  averages) changes what the model is trained on, so it is a modelling decision
  rather than a bug fix and should be made deliberately, with a before/after
  comparison.
- **The registry-alias deployment model** described under F4.
- **`Games.csv` has no recorded provenance.** No target fetches it, no doc says
  where it came from. See the cold-start section — needs one line from you.
