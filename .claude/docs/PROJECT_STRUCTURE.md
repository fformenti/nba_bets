# Project Structure

## The layering rules

**Libraries never have `__main__`; CLI modules never have logic.** Every entry
point is a thin argparse shell in `src/cli/`, one per Makefile target. Enforced
by `tests/test_imports_smoke.py`.

**One split definition.** `src/ml/datasets/splits.py::build_splits` decides which
games are train/validation/test. The sklearn path and the LLM dataset builder
both call it, so the two model families are always scored on the same gameIds.

**Two configs, two jobs.** `configs/features.yaml` decides which feature *tables*
get built (ETL). `configs/train/*.yaml` decides which of those columns a model
*consumes* (ML). Passing an experiment config where an ETL one belongs is a bug —
it used to break `predict-upcoming`.

**`winner` is the winning teamId, not a 0/1 flag.** Models predict 1 = home win.
`src/monitoring/scoring.py` does that conversion in one place.

## configs/

```
configs/
├── features.yaml                       # ETL: which feature tables to build (lags, alpha, beta)
├── train/
│   ├── _defaults.yaml                  # Shared base: splits, filters, model hyperparameters, ML feature set
│   ├── all_models.yaml                 # Overrides: sweep all four sklearn model families
│   ├── feature_audit.yaml              # Diagnostic: every feature group on, for Boruta-SHAP coverage
│   ├── llm_features.yaml               # Overrides: LLM feature set (adds win % and streak)
│   └── xgboost.yaml                    # The deployed model: all games, xgboost
├── train_llm/
│   └── llama31_8b_qlora.yaml           # QLoRA SFT; names the experiment whose splits it mirrors
└── predict/
    └── predict_classifier.yaml         # The deployed model URI
```

## src/

```
src/
├── cli/                                # EVERY entry point. One module per make target.
│   ├── ingest_raw_games.py             # make ingest-raw-games
│   ├── build_teams_history.py          # make build-teams-history
│   ├── build_teams_locations.py        # make build-teams-locations
│   ├── build_distances_table.py        # make build-distances-table
│   ├── build_polymarket_teams.py       # make build-polymarket-teams
│   ├── parse_league_schedule.py        # make parse-league-schedule
│   ├── process_ingested_games.py       # make process-ingested-games
│   ├── build_features.py               # make build-features
│   ├── fetch_league_schedule.py        # make fetch-league-schedule
│   ├── fetch_upcoming_games.py         # make fetch-upcoming-games
│   ├── fetch_game_results.py           # make fetch-game-results SOURCE=nba_api|placeholder
│   ├── append_game_results.py          # make append-game-results
│   ├── reconcile_postponed.py          # make reconcile-postponed
│   ├── retry_unresolved.py             # make retry-unresolved
│   ├── migrate_incremental_layout.py   # make migrate-incremental (one-time)
│   ├── predict_upcoming.py             # make predict-upcoming
│   ├── score_predictions.py            # make score-predictions
│   ├── train_classifier.py             # make train
│   ├── build_llm_dataset.py            # make build-llm-dataset
│   ├── train_llm.py                    # make train-llm
│   ├── evaluate_llm.py                 # make evaluate-llm
│   ├── place_bets.py                   # make bet-polymarket GAME_DATE=...
│   ├── build_game_slug_lookup.py       # make build-game-slug-lookup
│   ├── delete_experiment.py            # make delete-experiment EXPERIMENT=...
│   ├── delete_model.py                 # make delete-model ARGS=...
│   └── plot_home_win_ratio.py          # make plot-home-win-ratio
│
├── config/
│   ├── paths.py                        # All data path constants
│   ├── constants.py                    # Dates, team maps, neutral court labels
│   ├── secrets.py                      # require_env(): the only way to read a key
│   └── aws.py                          # AWS config
│
├── etl/
│   ├── make_features.py                # build_features(): the historical feature build
│   ├── process_ingested_games.py       # Split postponed vs played regular season
│   │
│   ├── ingestion/                      # raw → ingested
│   │   ├── raw_games.py                # Parse Games.csv
│   │   ├── parse_league_schedule.py    # Parse the schedule; drops All-Star by gameId
│   │   ├── append_games_results.py     # Append results to history, archive what is used
│   │   └── migrate_incremental_layout.py  # One-time move to the staged layout
│   │
│   ├── collectors/                     # External fetches
│   │   ├── league_schedule.py          # Re-pull the schedule (makeup dates)
│   │   ├── upcoming_games.py           # Next slate: what the schedule has and history lacks
│   │   ├── upcoming_games_results.py   # Route each game by status; count attempts
│   │   ├── postponed_watch.py          # Release parked/quarantined games
│   │   └── results/                    # PLUGGABLE outcome retrieval
│   │       ├── base.py                 # GameResult + GameStatus + ResultsSource protocol
│   │       ├── nba_api_source.py       # stats.nba.com (default)
│   │       └── placeholder_source.py   # TODO: stand-in reading manual_results/
│   │
│   ├── reference/                      # Slow-moving lookup tables
│   │   ├── teams_history.py            # Team/city/conference per season
│   │   ├── teams_locations.py          # Team GPS coordinates
│   │   ├── distances.py                # City-to-city distances
│   │   └── polymarket_teams.py         # teamId → Polymarket abbreviation
│   │
│   ├── features/                       # Feature table builders
│   │   ├── aggregator.py               # create_features_tables_from_config() + merge_features()
│   │   ├── winning_percentage.py       # Rolling win % (home/away/total)
│   │   ├── point_differential.py       # Rolling point differential
│   │   ├── east_vs_west.py             # Conference win/loss records
│   │   ├── rest_days.py                # Days since last game
│   │   ├── distances.py                # Rolling travel distance
│   │   ├── last_season_record.py       # Last season's record; SOS-adjusted variant
│   │   ├── streaks.py                  # Consecutive win/loss streak
│   │   ├── strength_of_schedule.py     # Rolling strength of schedule
│   │   ├── sos_adjusted_record.py      # SOS-adjusted winning percentage
│   │   ├── game_difficulty_score.py    # Per-game quality score
│   │   ├── playoff_standings.py        # Conference standings, GB, clinching flags
│   │   └── teams_arena.py              # Home arena lookup; derives neutral_court
│   │
│   ├── transformation/
│   │   └── add_conference.py           # Add conference columns
│   │
│   └── utils/
│       └── common.py                   # read_json, save_feature_table, get_nba_season, …
│
├── ml/
│   ├── README.md                       # Modelling guide: the two invariants, training, LLM
│   │
│   ├── config/
│   │   ├── schema.py                   # Pydantic configs (Experiment, Prediction, LLM)
│   │   └── loader.py                   # YAML → config objects, with includes
│   │
│   ├── datasets/
│   │   ├── splits.py                   # build_splits(): THE split definition
│   │   ├── loaders.py                  # Load games_features.csv
│   │   └── splitters.py                # season_split (the default) / temporal / random
│   │
│   ├── features/
│   │   ├── engineering.py              # Delta features, conference features, column resolution
│   │   ├── preprocessing.py            # Scaling, imputation, outlier handling
│   │   └── selection.py                # Boruta-SHAP feature selection
│   │
│   ├── models/
│   │   ├── trainer.py                  # ModelTrainer (train, evaluate, CV)
│   │   ├── registry.py                 # ModelRegistry (local persistence)
│   │   └── baseline.py                 # Naive baseline models
│   │
│   ├── training/
│   │   ├── classifier.py               # train_classifier(): one experiment end to end
│   │   ├── experiment.py               # train_single_model(): the training body
│   │   ├── data_prep.py                # Load, validate, prepare
│   │   ├── model_factory.py            # Build models from config
│   │   ├── runners.py                  # Baseline and configured-model runners
│   │   └── weighting.py                # THE sample-weight definition (within-season ramp)
│   │
│   ├── llm/                            # The LLM is a second ENCODING of the same experiment
│   │   ├── serialization.py            # serialize_row(): the ONLY table→text boundary
│   │   ├── dataset.py                  # build_llm_dataset() from build_splits()
│   │   ├── finetune.py                 # QLoRA SFT: preflight, Hub resume, SFTTrainer
│   │   ├── train.py                    # Run orchestration + tracking
│   │   ├── evaluate.py                 # Score an adapter as a sign classifier
│   │   └── testers.py                  # RegressionTester / ClassificationTester
│   │
│   ├── prediction/
│   │   ├── pipeline.py                 # run_prediction_pipeline() + upsert_predictions()
│   │   ├── features.py                 # Inference-time feature construction
│   │   └── io.py                       # Load upcoming-game JSON
│   │
│   ├── evaluation/
│   │   ├── metrics.py                  # Classification metrics, Brier, ECE
│   │   ├── visualization.py            # Confusion matrix, ROC, calibration, SHAP
│   │   └── analysis.py                 # Prediction error pattern analysis
│   │
│   ├── tracking/
│   │   ├── mlflow_tracker.py           # MLflowTracker context manager
│   │   ├── delete_experiment.py        # Ops: delete experiments
│   │   └── delete_model.py             # Ops: delete registered models
│   │
│   └── utils/
│       ├── validation.py               # Data validation helpers
│       └── shap.py                     # SHAP output compatibility
│
├── betting/                            # Polymarket — consumes model output, not modelling
│   ├── slugs.py                        # slugify(), get_game_slug()
│   ├── polymarket_client.py            # Gamma/CLOB HTTP; MarketSide; get_market_prices() seam
│   ├── sizing.py                       # Edge-proportional stake sizing
│   ├── bets.py                         # Build and save a daily buying strategy
│   └── game_slugs.py                   # gameId → slug lookup for the test seasons
│
├── monitoring/
│   └── scoring.py                      # score_predictions(): live accuracy vs played games
│
├── eda/
│   └── home_win_ratio_by_season.py     # Home win ratio bar chart by season
│
└── utils/
    └── logging_config.py               # Shared logging setup
```

## Everything else

- `data/` (gitignored) — `raw/historical/` holds the source `Games.csv` and the
  league schedule. `raw/incremental/` gives every collected game one directory
  per state, so "where is this file?" answers "what happened to it?":
  `upcoming_games/` (pending, no terminal result yet), `upcoming_games_results/`
  (final score fetched, not yet in history), `archive/results/` (consumed),
  `postponed/` (parked, watched for a makeup date), `unresolved/` (quarantined —
  the source never settled it), and `manual_results/` (hand-dropped outcomes for
  the placeholder source). `ingested/` holds exactly one table,
  `games_updated_history.csv`, the record every downstream step reads:
  `ingest-raw-games` merges the raw archive into it and `append-game-results`
  adds the results inbox to it.
  `processed/` holds the feature tables, `games_features.csv` and the
  Polymarket slug lookup. The intermediate
  feature tables exist twice under the same filenames:
  `processed/regular_season/features/` is the ETL's, built from all of history by
  `build-features`; `processed/prediction/features/` is prediction's, built from
  one season plus the slate. They are separate because sharing one directory left
  the ETL's tables truncated after every prediction run. `predictions/`
  holds `upcoming_games_predictions.csv` (one row per game, deduped on gameId),
  the accuracy scorecard, the per-game scored file and the daily bet strategies.
- `tests/` — feature-level unit tests plus the invariant tests:
  imports smoke (every module imports; no `__main__` outside `src/cli/`), LLM
  split parity, LLM serialization, prediction upsert, prediction scoring, the
  results-source contract, upcoming-game selection (the queue cannot jam), the
  incremental append, and feature-table isolation (the ETL and prediction never
  share an output directory).
- `dead_code/` — code removed because nothing referenced it, kept rather than
  deleted. See its README.
- `outputs/` (gitignored) — training artifacts: per-experiment plots, CV results,
  feature selection, and LLM evaluation charts.
- `docs/` — `RUNPOD_TRAINING.md`, GPU box setup for LLM fine-tuning.
- `sandbox/` (gitignored) — exploratory notebooks.
- `mlruns/`, `mlflow.db` (gitignored) — MLflow tracking.
- `.claude/` — project settings, hooks, skills, and this doc.

## Pipelines

```
Historical (make historical-etl / full-rebuild):
  Games.csv → ingest-raw-games → process-ingested-games → build-features
                                                            ↓
                                                 games_features.csv

The daily loop (make daily-cycle):
  fetch-league-schedule  stats.nba.com → LeagueSchedule25_26.csv
  parse-league-schedule  → league_schedule.csv
  reconcile-postponed    parked games with a new date → released
  fetch-upcoming-games   schedule minus history → upcoming_games/
  predict-upcoming       → upcoming_games_predictions.csv  (deduped)
        ⋯ games are played ⋯
  fetch-game-results     ResultsSource → by status:
                           FINAL     → upcoming_games_results/
                           POSTPONED → postponed/
                           otherwise → stays pending, attempts++
                                       → unresolved/ at MAX_FETCH_ATTEMPTS
  score-predictions      predictions ⋈ outcomes → prediction_scorecard.csv + MLflow
  append-game-results    → games_updated_history.csv, consumed → archive/results/
  process-ingested-games
  build-features         history now includes the games just played
```

Two invariants hold this together. **A game is unresolved when the schedule
knows it and history does not** — that is the whole selection rule, and it is
why a makeup game rescheduled into the past is still reachable. **Postponement
is read from the source's status, never inferred from a 0-0 scoreline** — a game
that has not tipped off looks identical to one that was called off, and guessing
wrong writes a fixture into history that never happened.

Run the loop without a live results feed using
`make daily-cycle SOURCE=placeholder`, after dropping `{gameId}.json` files into
`data/raw/incremental/manual_results/`. Set `"status"` in those files to
exercise the postponed and still-pending paths.
