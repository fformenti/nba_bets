# outputs/: layout, result schema, plotting

Everything the project writes that is not data (`data/`) or a model (`mlruns/`,
`models/`) lands under `outputs/`, which git ignores. This doc is the contract
for what goes where. Every path in it is declared once in `src/config/paths.py`
and nowhere else.

## The two rules

1. **Training, evaluation and EDA write result files. Only the plotting module draws.**
   `make train`, `make inspect-llm` and `make evaluate-llm` write per-game
   predictions, metrics and metadata. `src/eda/` writes tables. `make plot`,
   `make plot-compare` and the EDA figures read those files and draw. No
   training, evaluation or EDA code imports `src/ml/plotting/`.
2. **Every model family writes the same per-game schema.** The sklearn
   classifiers, the margin regressor and the LLM adapters all write one row per
   game per (model, calibration) in the format below. That shared format is
   what lets one chart overlay xgboost, a regressor, an LLM adapter and the
   Polymarket price.

## Layout

```
outputs/
├── runs/<run_id>/                  one folder per training or evaluation run (primary results)
│   ├── run.json                    what the run is: family, task, config, git sha, MLflow id, selected series
│   ├── predictions/<split>.csv     per-game predictions, shared schema (split = validation | test)
│   ├── metrics/<split>.json        score_predictions() of that file, one row per (model, calibration)
│   ├── model/                      sklearn only: what the shipped model says about its features
│   │   ├── cv_results/<model>.csv      hyperparameter search results
│   │   ├── feature_selection.json      Boruta-SHAP results (BorutaShapSelector.to_dict)
│   │   ├── feature_importance.csv      feature, importance
│   │   └── shap.csv                    game_id, feature, shap_value, feature_value (≤ 500 test games)
│   ├── llm/                        LLM only: per-game diagnostics no other family has
│   │   ├── inspection_<split>.csv      raw decoded text, P(+), P(-), expected margin, top tokens
│   │   └── summary_<split>.json        diagnostics.summarize() of the inspection CSV
│   ├── tracking/mlflow.db          LLM only: the Colab session's MLflow database, copied by train_llm.ipynb
│   └── report/                     DERIVED by `make plot`; safe to delete and redraw
│       ├── figures/*.png|*.html
│       └── tables/*.csv
├── comparisons/<name>/             DERIVED by `make plot-compare`: several runs on shared games
│   ├── figures/                    roc_curve, calibration_curve, season_phase_accuracy
│   └── tables/scoreboard.csv       every series scored on the shared games
├── eda/<topic>/
│   ├── tables/*.csv                the EDA result (src/eda/ writes it)
│   └── figures/*.png               drawn from those tables by src/ml/plotting/eda.py
├── checkpoints/<run_name>/         LLM training state (checkpoint-N); large, regenerable from the Hub
└── archive/<date>_<name>/          outputs from before this layout, kept as they were, never plotted
```

**Why runs are the top-level split:** the run is the unit you create, copy (`scp`
from a GPU box, Drive from Colab), compare and delete. Task type and split are
properties of rows, not folders. `run.json` records the task, and `split` is a
column. A regressor's winner calls can therefore sit on the same chart as a
classifier's, and one run's validation and test files stay together.

**Run IDs:**
- **sklearn (`make train`):** `<config stem>-<YYYYmmdd-HHMMSS>`, e.g.
  `xgboost-20261023-165536`. The MLflow run carries the same name. Runs
  accumulate; nothing is overwritten.
- **LLM:** the training run name, which is also the Hub repo (e.g.
  `e4-llama32-1b`). Re-inspecting a split replaces that split's series inside
  the same folder.

## Result schema

### `predictions/<split>.csv`

One row per game per (model, calibration). Defined and validated in
`src/ml/results/schema.py`.

| Column | Type | Meaning |
|---|---|---|
| `run_id` | str | the folder name |
| `model` | str | sklearn: the family (`xgboost`, `record_baseline`, …). LLM: the run name, or `<run>-greedy` for `evaluate-llm`'s decoded answers |
| `calibration` | `none` / `sigmoid` / `isotonic` / `platt` | `none` is the model's own output. The other values name the calibrator applied on top of it |
| `split` | `validation` / `test` | |
| `game_id` | int | the `gameId` of the game tables |
| `y_true` | 0/1 | 1 = the home team won. (The game tables' `winner` is a teamId; this column is not.) |
| `y_pred` | 0/1 | the hard call, 1 = home. Classifier: `predict`. Regressor: margin > 0. LLM: `p_home > 0.5` |
| `p_home` | float or empty | P(home win). Empty for a model with no probability (regressors, threshold baselines, greedy decoding) |
| `pred_margin` | float or empty | predicted home − away, in real points. Regressor on `margin`; LLM's expected margin × `margin_scale` |
| `true_margin` | float or empty | actual home − away points |

Which series each writer emits:

| Writer | Series |
|---|---|
| `make train` (classifier) | every family × {validation, test}, `calibration=none`. The selected family also gets `sigmoid` and `isotonic`. On validation these are the **out-of-fold** probabilities the calibrator was chosen on; on test, the refit calibrator. Test also gets `record_baseline`, `point_differential_baseline` (hard calls only) and `home_win_baseline` |
| `make train` (regressor, `target_column: margin` or `margin_scaled`) | every family × {validation, test} plus `mean_baseline` on test. `pred_margin` is filled only for `margin`: a `margin_scaled` prediction is in other units |
| `make inspect-llm` | `<run>` × `none` (raw first-token probability) and `platt`. Platt is cross-fitted in two seeded halves on validation; on test it is fitted on the run's validation |
| `make evaluate-llm` | `<run>-greedy` × `none` (decoded answers; no probability) |

Polymarket is **never** written into a run. It is external data, and a benchmark
rather than a model output. The plotting module joins it by `gameId` at draw
time (`src/ml/evaluation/polymarket_baseline.py`), restricted to the games both
sides cover.

Game metadata (season, teams, conference, overtimes, neutral court, score, games
played) is not stored per run either. These are facts about games, so the
plotting module reads them from `data/processed/regular_season/games_features.csv`
by `gameId`. Without that table, the breakdown figures are skipped.

### `metrics/<split>.json`

```json
{"schema_version": 1, "run_id": "...", "split": "validation",
 "rows": [{"model": "xgboost", "calibration": "none", "n_games": 2949,
           "accuracy": 0.648, "home_pick_rate": 0.62, "home_win_rate": 0.55,
           "brier_score": 0.219, "log_loss": 0.628, "ece": 0.02, "roc_auc": 0.698,
           "mae": null, "rmse": null}]}
```

Produced by `src/ml/results/scoring.py::score_predictions`, the one scoring
function for every family. It is rewritten whenever a predictions file
changes. Probability metrics need every row of a series to carry `p_home`;
`mae`/`rmse` need both margins. Log-loss clips at 1e-6, matching
`diagnostics.summarize`.

### `run.json`

| Field | Meaning |
|---|---|
| `run_id`, `family` (`sklearn`/`llm`), `task` (`classification`/`regression`) | |
| `selected_model`, `selected_calibration` | the series the run stands for: the family that ships, with the calibrator only if it was adopted. LLM runs select `platt` once validation is scored, because the scoreboard reads an adapter after Platt |
| `best_calibrator` | the calibrator validation preferred, adopted or not. The calibration figure draws it against `none` |
| `config_path`, `git_commit`, `created_at` | |
| `mlflow_run_id`, `mlflow_tracking_uri` | `make plot` attaches `report/` to this MLflow run |
| `splits` | `{split: {"n_games": int}}` |
| `extra` | family-specific: `n_features`, `split_method`, `hub_model_id`, `base_only`, `few_shot`, `migrated_from`, … |
| `schema_version` | 1 |

## Commands

```bash
make train TRAIN_CONFIG=xgboost        # writes outputs/runs/xgboost-<ts>/, then draws it (make plot RUN=latest)
make plot RUN=<run_id> [SPLIT=validation]
make plot RUN=latest                   # most recent run.json by created_at, any family
make plot-compare NAME=llm_vs_xgb SERIES="xgboost-20261023-165536 e11-llama32-1b-compact"
make plot-compare NAME=cal SPLIT=test SERIES="xgboost-20261023-165536:xgboost:none xgboost-20261023-165536:xgboost:sigmoid"
make plot-compare NAME=vs_market SPLIT=test MARKET=1 SERIES="xgboost-20261023-165536"
make plot-compare NAME=e17_vs_record RECORD=1 SERIES="e17-llama32-1b-compact-min4"
make plot-home-win-ratio               # writes eda/home_win_ratio/tables/home_win_ratio_by_season.csv, then the bar chart
make plot-home-win-ratio-line          # same table, then the line chart (seasons >= 1980/81)
```

A series is `run_id[:model[:calibration]]`. Omitted parts default to the run's
`selected_model` and `selected_calibration`. Every series in a comparison is
restricted to the gameIds all of them scored, and to the games Polymarket
priced when `MARKET=1` (Polymarket only covers the test seasons, 2024/25 on).
`RECORD=1` adds the record baseline on those same games: the team with the better
season-to-date record (`record_L82`), ties to home, read from the game table. It
is a hard call, so its scoreboard row has accuracy only. Comparing each run with
the benchmarks on its own games is how runs with different game sets are compared.
Titles carry the game count. When a raw series and its calibrated twin are both
given, the calibration figure draws the raw one dashed.

From Python: `src.ml.results.read_predictions(run_id, ["validation"])` returns the
frame; `src.ml.plotting.report.build_run_report` / `build_comparison` are what
the CLIs call.

### What `make plot` draws

Each figure is drawn only when its inputs exist; a figure that cannot be drawn
is skipped with a warning.

| Figure (`report/figures/`) | Needs |
|---|---|
| `confusion_matrix` | any series |
| `roc_curve`, `prediction_accuracy_by_bin` | `p_home` |
| `calibration_curve` | `p_home` on the `none` series (+ `best_calibrator` series) |
| `error_by_team`, `accuracy_by_season`, `conference_matchup_accuracy`, `overtime_accuracy`, `neutral_court_accuracy`, `season_phase_accuracy` (+ record baseline), `score_margin_calibration` | the game table; each also writes `tables/<name>.csv` |
| `boruta_shap_selection`, `feature_importance`, `shap_summary` | `model/` files |
| `roc_curve_vs_market`, `calibration_curve_vs_market`, `season_phase_accuracy_vs_market`, `tables/polymarket_baseline.csv` | games Polymarket priced (test seasons) |
| `predictions_vs_actual`, `residuals` | regression runs |
| `accuracy_trend.html`, `confidence_accuracy.html`, `scatter.html` | LLM runs (raw series) |
| `tables/metrics.csv` | always: `score_predictions` for the split |

## Code map

| Concern | Module |
|---|---|
| every output path | `src/config/paths.py` (`RUNS_DIR`, `RunPaths`, `COMPARISONS_DIR`, `EDA_OUTPUTS_DIR`, `CHECKPOINTS_DIR`, `ARCHIVE_OUTPUTS_DIR`) |
| schema, validation, `RunInfo` | `src/ml/results/schema.py` |
| read/write run files | `src/ml/results/store.py` |
| metrics from predictions | `src/ml/results/scoring.py` |
| sklearn writer | `src/ml/training/experiment.py`, `src/ml/training/regressor.py` |
| LLM writer | `src/ml/llm/results.py` (from `inspect_llm`), `src/ml/llm/evaluate.py` |
| SHAP values (compute) | `src/ml/utils/shap.py::compute_shap_frame` |
| breakdown tables (compute) | `src/ml/evaluation/analysis.py` |
| figures | `src/ml/plotting/charts.py`, `breakdowns.py`, `llm.py`, `eda.py` |
| EDA tables | `src/eda/` (`home_win_ratio_by_season.py`: season, n_games, home_wins, home_win_ratio) |
| run report / comparisons | `src/ml/plotting/report.py`; CLIs `src/cli/plot_run.py`, `src/cli/plot_compare.py` |
| one-time migration | `src/ml/results/migrate.py`; CLI `src/cli/migrate_outputs_layout.py` |

## Migration from the old layout (2026-10-23)

`make migrate-outputs` (with `ARGS=--dry-run` to preview) moved every file. It
deleted nothing except `.DS_Store`. All 139 original files were checksum-verified
at their new paths.

| Old | New |
|---|---|
| `all_models/**` | `archive/2026-10-17_all_models/**` |
| `xgboost/**` | `archive/2026-08-18_xgboost/**` |
| `feature_audit/**` | `archive/2026-08-12_feature_audit/**` |
| `opus_search/**` (its config is deleted) | `archive/2026-08-13_opus_search/**` |
| `tuning/**` (its harness is deleted) | `archive/2026-08-12_season_decay_tuning/**` |
| `enriched_games/viz/*.png` | `eda/home_win_ratio/figures/*.png` (now drawn from `eda/home_win_ratio/tables/home_win_ratio_by_season.csv`; pixel-identical to the old code's output) |
| `llm/eval/<run>/inspection_<split>.csv` | `runs/<run>/llm/inspection_<split>.csv`, **plus** converted `predictions/`, `metrics/`, `llm/summary_*.json`, `run.json` |
| `llm/eval/<run>/mlflow.db` | `runs/<run>/tracking/mlflow.db` |
| `llm/eval/scoreboard_*.csv` (Drive only) | `comparisons/llm_scoreboard/tables/` |
| `llm/qlora/<model>/<run>/` (checkpoints; none existed locally) | `checkpoints/<run>/` |

**Converted LLM runs reproduce the handoff scoreboard.** Validation accuracy and
log-loss after Platt match `docs/plans/llm-finetune-handoff.md` §2 for all nine runs.
That table's AUC column mixes raw and Platt-cross-fitted AUC, so it differs from
`metrics/validation.json` (raw `none` AUC = `diagnostics.summarize`'s
`auc_p_home`) by up to 0.003.

**Archived sklearn outputs cannot be redrawn.** Those runs saved figures and
aggregate tables, never per-game predictions. New runs regenerate the same
figure set: a fresh `make train TRAIN_CONFIG=xgboost` produced every figure
type the archive holds.

**The Drive mirror** that the Colab notebooks read (`MyDrive/nba_bets/outputs/`)
still has the old `llm/eval/` layout. Migrate it once with:

```bash
uv run python -m src.cli.migrate_outputs_layout --outputs-dir "<Drive mount>/nba_bets/outputs"
```

## Adding a new model family or figure

- **New family:** build its per-game frame with
  `src.ml.results.build_predictions`, write it with `write_predictions(frame,
  run_id, split)` and describe the run with `write_run_info(RunInfo(...))`. Use
  an existing `family`/`task` or extend the tuples in `schema.py`. Nothing in
  plotting needs to change.
- **New figure:** add a function to `src/ml/plotting/` that takes arrays or
  tables, and call it from `report.py`. If it needs a quantity no file holds,
  add that quantity to a writer's output. Don't compute model internals at plot
  time.
- **New EDA topic:** a function in `src/eda/` writes `outputs/eda/<topic>/tables/*.csv`;
  a function in `src/ml/plotting/eda.py` draws from that file; the CLI calls one
  after the other. Declare both paths in `src/config/paths.py`.
- **New output location:** declare it in `src/config/paths.py` and add it to
  the layout above.
