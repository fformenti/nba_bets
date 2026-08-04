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
│   ├── _defaults.yaml                  # Shared training defaults
│   ├── train_all.yaml                  # Overrides: all games
│   ├── train_different.yaml            # Overrides: cross-conference
│   └── train_same.yaml                 # Overrides: same conference
├── train_llm/
│   └── llama31_8b_qlora.yaml           # QLoRA SFT; names the experiment whose splits it mirrors
└── predict/
    └── predict_classifier.yaml         # Model URIs per conference filter
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
│   ├── build_holdout_set.py            # make build-holdout-set (run once)
│   ├── fetch_upcoming_games.py         # make fetch-upcoming-games
│   ├── fetch_game_results.py           # make fetch-game-results SOURCE=nba_api|placeholder
│   ├── append_game_results.py          # make append-game-results
│   ├── predict_upcoming.py             # make predict-upcoming
│   ├── score_predictions.py            # make score-predictions
│   ├── train_classifier.py             # make train
│   ├── run_experiments.py              # make train-all
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
│   │   ├── parse_league_schedule.py    # Parse the league schedule CSV
│   │   └── append_games_results.py     # Merge played results into history
│   │
│   ├── collectors/                     # External fetches
│   │   ├── upcoming_games.py           # Next slate from the processed schedule
│   │   ├── upcoming_games_results.py   # Stamp outcomes onto game payloads
│   │   └── results/                    # PLUGGABLE outcome retrieval
│   │       ├── base.py                 # GameResult + ResultsSource protocol
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
│   │   ├── holdout.py                  # Freeze the holdout (run once)
│   │   ├── loaders.py                  # Load games_features.csv
│   │   └── splitters.py                # Temporal / random / fixed-holdout primitives
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
│   │   └── runners.py                  # Baseline and configured-model runners
│   │
│   ├── llm/                            # The LLM is a second ENCODING of the same experiment
│   │   ├── serialization.py            # serialize_row(): the ONLY table→text boundary
│   │   ├── dataset.py                  # build_llm_dataset() from build_splits()
│   │   ├── prompts.py                  # Legacy prose template (lazy tokenizer)
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
│   └── game_slugs.py                   # gameId → slug lookup for the holdout
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

- `data/` (gitignored) — `raw/historical/` holds the source `Games.csv`;
  `raw/incremental/` holds `upcoming_games/` (fetched, not yet played),
  `upcoming_games_results/` (played, enriched) and `manual_results/`
  (hand-dropped outcomes for the placeholder source). `ingested/` holds
  `games_updated_history.csv`, the durable record of every played game.
  `processed/` holds the feature tables, `games_features.csv`, the frozen
  `holdout/test_metadata.csv` and the Polymarket slug lookup. `predictions/`
  holds `upcoming_games_predictions.csv` (deduped on gameId + conference_filter),
  the accuracy scorecard, the per-game scored file and the daily bet strategies.
- `tests/` — feature-level unit tests plus the invariant tests:
  imports smoke (every module imports; no `__main__` outside `src/cli/`), LLM
  split parity, LLM serialization, prediction upsert, prediction scoring, and
  the results-source contract.
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
  fetch-upcoming-games   league schedule → upcoming_games/
  predict-upcoming       → upcoming_games_predictions.csv  (deduped)
        ⋯ games are played ⋯
  fetch-game-results     ResultsSource → upcoming_games_results/
  score-predictions      predictions ⋈ outcomes → prediction_scorecard.csv + MLflow
  append-game-results    → games_updated_history.csv
  process-ingested-games
  build-features         history now includes the games just played
```

Run the loop without a live results feed using
`make daily-cycle SOURCE=placeholder`, after dropping `{gameId}.json` files into
`data/raw/incremental/manual_results/`.
