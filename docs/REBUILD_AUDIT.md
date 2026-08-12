# Full-rebuild audit — 2026-08-11

> **Every model metric below is superseded.** They were selected on the test set
> — the calibration method by test Brier and the model family by test accuracy —
> and they come from the three-model conference split, which has since been
> collapsed to one model (docs/CONFERENCE_SPLIT.md) and retrained without
> season-decay weighting. Both selections now happen on validation. The findings
> and the reproducibility results stand; the accuracy figures do not.
>
> The `conference_filter` machinery those three models ran on is gone as of
> 2026-08-12, so the per-filter columns below describe a pipeline that no longer
> exists. `games_features.csv` is now 66,046 rows × 228 columns: the three
> `games_played_at_east` / `_at_west` / `_east_vs_west` columns were computed,
> merged and then dropped unread by every code path, and are no longer returned
> by the east/west builder.

Every derived file under `data/` was deleted and the pipeline driven back to its
prior state from the tracked inputs alone: parse → features → holdout → train →
two full daily cycles (2026-01-25 and 2026-01-26 slates) → score → append →
rebuild. This records what reproduced, what did not, and what the run exposed.

The audit run itself changed no source. **Every finding below has since been
fixed — see "Resolution" at the end for what changed and how it was verified.**

## What reproduced exactly

| Artefact | Result |
|---|---|
| `data/ingested/games_updated_history.csv` | **byte-identical** to the pre-wipe file (72,641 rows) |
| `data/raw/incremental/archive/results/*.json` | identical, all 13 games |
| `games_features.csv` | 66,046 rows × 231 columns, no all-NaN columns |
| Trained models | **bitwise identical** — old and new runs give the same probabilities on the same input, for all three conference filters |
| Test accuracy | 0.6625 / 0.6199 / 0.6488 (same / different / all) — matches the previous run to 4 dp |
| MLflow params | 0 differences across all three model pairs |
| Scorecard accuracy | 0.6154 overall, 0.5 different, 0.7143 same — identical |
| `make test` | 320 passed |

The ETL and training halves are deterministic. `data/raw/reference/` and
`data/raw/historical/handmade/` being in git (commit `0a21ff9`) is what made this
possible with no credentials and no API spend.

One number in the plan was wrong: history is **72,628** rows after
`ingest-raw-games`, not 72,629. `Games.csv` row `gameId 12500014` is a preseason
game (`gameLabel: Preseason`) and is filtered by design.

## Findings, ranked

### F1 — Upcoming-game placeholders corrupt serving-time features (high)

`build_prediction_feature_base` concatenates the slate onto history and builds the
feature tables from the combined frame. But `fix_upcoming_games_cols` fills the
slate rows with placeholders — `homeScore = awayScore = 0`, `winner = 0`, and no
location columns at all. Any feature that aggregates over *other* games then reads
those fakes as real.

Measured by rebuilding the prediction-time tables and diffing against the ETL's
values for the same gameIds, after the games were played and appended:

- **`distance_L1/L3/L7/L14_VT` are 0 for every predicted game.** The true values
  for the 2026-01-25 slate were 145 – 2,796 miles. Cause:
  `enrich_games_locations` is called only in `src/etl/ingestion/raw_games.py`, so
  slate rows reach `make_teams_distances_table_season` with
  `hometeamLocation` / `awayteamLocation` / `gameLocation` NaN, and the unknown-location
  branch records 0. The home team's 0 is correct by coincidence, which is why
  only the `_VT` side shows up.
  `distance_L1_delta` and `distance_L3_delta` are selected features of the
  **same** model; `distance_L14_delta` of the **all** model. In training they
  carry real travel; in production they are identically zero.
- **`sos_*` and `sos_adj_record_*` differ by up to 0.014** on slate rows.
- **`clinched_playoff_berth`, `clinched_final_seed`, `indifference_flag`** flip on
  one row per slate. `indifference_flag_delta` is selected by two of three models.

The two-season history window is **not** the cause and is sound: for the 3,740
*historical* rows in the prediction frame, SOS matches the ETL exactly at every
lag — 0 differing values. Only the slate's own rows diverge.

Impact on the shipped predictions for these 13 games, re-scoring the same models
against the ETL's (correct) feature values: **mean |Δp| = 0.011, max |Δp| = 0.071,
0 predicted-class flips.** Small on this slate, but it is a permanent bias in the
direction of "nobody travelled", and it scales with how much the model leans on
the distance features.

This also explains the one pre/post difference in `upcoming_games_predictions.csv`:
identical gameIds, identical predicted classes, probabilities differing by up to
0.063. With models and history byte-identical, the residual is serving-time
feature values, which depend on the exact frame at predict time.

### F2 — `--promote` corrupts the config it writes, and half-deploys (high)

`_update_predict_config` (`src/ml/training/classifier.py:129`):

```python
pattern = rf'(?ms)^model_uris:.*?^(\s+{conference_filter}:\s+")([^"]*)(")'
content, n = re.subn(pattern, rf"\g<1>{model_uri}\3", content, count=1)
```

`^model_uris:.*?` is inside the match but not inside any capture group, so the
replacement drops that line. Observed, in one `run_experiments --promote`:

1. `same` is rewritten and the `model_uris:` key is **deleted**;
2. `different` and `all` no longer find the anchor, log a warning, and no-op —
   they keep pointing at the *previous* run's models;
3. the three URIs, now dangling at two-space indent, get absorbed into the
   preceding `data:` block, so `model_uris` is absent from the parsed config
   entirely.

`make train PROMOTE=1` is the sanctioned deploy path, and it silently produces a
config in which two of three models are stale and the third is unreachable. The
next `predict-upcoming` fails on the missing-URI guard — loud, but only after the
fact.

### F3 — A fresh clone cannot get past step 4 (medium)

`src/etl/process_ingested_games.py:103` writes
`data/processed/regular_season/games.csv` without creating the directory. It is
the only writer in `src/` missing `mkdir(parents=True, exist_ok=True)` — 33 other
write sites have one. The rebuild died here and needed a manual `mkdir`.

### F4 — The train/serve guard cannot see F1 (medium)

`_align_features` raises when a feature is NaN for every row of a slate. F1
produces features that are *wrong*, not missing — `0.0` for distance, a shifted
float for SOS — so the guard stays silent. It was silent for both slates.

Compounding it, `allow_missing_features: true` in
`configs/predict/predict_classifier.yaml` downgrades even the checks that do fire
to warnings. Nothing in the run's logs flagged anything.

The 320-test suite also passes with F1 live: there is no test that compares a
feature computed by the prediction path against the same feature computed by the
ETL. That comparison is cheap and mechanical — it is how F1 was found.

### F5 — A run's feature set is not recoverable from MLflow params (low)

Only counts are logged (`n_features`, `feature_selection_n_confirmed`, …), not the
selected column names. Confirming that two runs selected the same features
required loading both model artifacts and reading `feature_names_in_`.

### F6 — The two feature paths emit different column sets (low)

The prediction path produces `pts_diff_HT/VT/...` and `win_bool_HT/VT/...` that
the ETL path does not; the ETL produces `neutral_court`, `pts_diff`, `win_bool`
and the three location columns that the prediction path does not. Harmless today
because no model selects them, but it is the same class of divergence as F1 and
nothing prevents a future feature from landing on the wrong side of it.

## Still open from the previous audit

- **Model URIs are hardcoded run IDs.** A registered-model alias
  (`models:/nba_classification_xgboost_same@champion`) would make retraining
  self-wiring and would remove the fragile regex in F2 entirely.
- **`compute_playoff_flags` is ~80% of the 2.5-minute feature build**
  (`_compute_sos_columns` is a further ~24%). Not touched; measured, not guessed.
- **`Games.csv` has no recorded provenance.** No target fetches it, no doc says
  where it came from. One line in the `ingest-raw-games` docstring closes the last
  cold-start gap.

## Resolution — 2026-08-12

All six fixed, plus a seventh the fixes themselves turned up.

| # | Fix | Where |
|---|---|---|
| F1a | Slate rows are enriched with locations before the distance tables are built | `src/ml/prediction/features.py` |
| F1b | A `has_result` predicate; result-less games no longer contribute to anyone's cumulative win percentage | `src/etl/utils/common.py`, `src/etl/features/strength_of_schedule.py` |
| F1c | Season length comes from the parsed league schedule, not from the games present | `src/etl/features/playoff_standings.py` |
| F2 | The `model_uris:` anchor is captured and re-emitted; the rewrite is re-parsed and verified before being written | `src/ml/training/classifier.py` |
| F3 | `atomic_write_csv`, which creates the directory and writes via a temp file | `src/etl/process_ingested_games.py` |
| F4 | Four parity regressions on a synthetic mini-league, needing nothing from `data/` | `tests/test_train_serve_parity.py` |
| F5 | Selected / confirmed / tentative / rejected feature names logged as a JSON artifact | `src/ml/training/experiment.py` |
| F6 | `win_bool`, `pts_diff` and `neutral_court` set on the slate so both paths emit the same columns | `src/ml/prediction/features.py` |

### F7 — SOS read opponents' records at tip-off time, not game day

Found by the new parity test, and the reason F1b alone was not enough. Two games
on the same evening tip off hours apart, so a timestamp comparison let the later
one read the earlier one's result. The prediction pipeline can never do that: it
predicts the whole slate before any of it is played. Training was consuming
information serving cannot have.

Opponent strength is now evaluated as of the **start of the game's day** — the
convention `compute_playoff_flags` already used, for the same reason. This is the
broadest change here: it shifts `sos_*` on ~25,000 of 66,046 rows, and the derived
`sos_adj_record_*` / `gds_*` on ~37,000, by a mean of 0.003–0.006 each.

### Verified

- **Parity: 93 mismatching columns → 0**, same script, same two slates
  (2026-01-25, 2026-01-26). The column-set asymmetry is gone too.
- `distance_L1_VT` for the 01-25 slate now reads 145–2,796 miles at prediction
  time and matches the ETL exactly, against 0.0 for every game before.
- **2025/26 playoff-berth clinches: 138 → 0.** No team clinches in December.
- `--promote` wrote all three URIs, with `model_uris` intact and the `data:`
  block clean — the direct F2 regression.
- Holdout accuracy, before → after: same **0.6625 → 0.6662**, different
  **0.6199 → 0.6223**, all **0.6488 → 0.6519**. Scorecard accuracy over the 26
  live predictions is unchanged at 0.6154, with Brier 0.2460 → 0.2472 and ECE
  0.1202 → 0.1010.
- 324 tests pass, against 320 before.

### Still open

Unchanged from the previous audit and deliberately not touched here: the
registry-alias deployment model, `compute_playoff_flags` at ~80% of the feature
build, and `Games.csv` provenance.
