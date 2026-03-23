# Project Structure

```
nba_bets/
├── configs/                                    # YAML configuration files
│   ├── ingestion/
│   │   └── incremental_ingestion.yaml          # Incremental ingestion config
│   ├── train/
│   │   ├── train_all.yaml                      # Training config: all features
│   │   ├── train_different.yaml                # Training config: different splits
│   │   └── train_same.yaml                     # Training config: same splits
│   └── predict/
│       └── predict_classifier.yaml             # Prediction config
│
├── data/                                       # All data (gitignored)
│   ├── raw/
│   │   ├── historical/
│   │   │   ├── games/                          # Source Games.csv
│   │   │   └── handmade/                       # Manual reference data (league schedule, etc.)
│   │   └── incremental/
│   │       ├── upcoming_games/                 # Fetched upcoming games (CSVs)
│   │       └── upcoming_games_results/         # Fetched game results (CSVs)
│   ├── ingested/
│   │   ├── historical/                         # Parsed & filtered historical games
│   │   └── incremental/                        # Parsed incremental games
│   ├── processed/
│   │   └── regular_season/
│   │       └── features/                       # Feature tables + games_features.csv
│   └── predictions/
│       └── daily_bets/                         # Prediction outputs
│
├── src/
│   ├── config/
│   │   ├── paths.py                            # All data path constants
│   │   ├── constants.py                        # Dates, team maps, neutral court labels
│   │   └── aws.py                              # AWS config
│   │
│   ├── data_creation/
│   │   └── polymarket_teams_abrev.py           # Team abbreviation mapping for Polymarket
│   │
│   ├── etl/
│   │   ├── ingestion/
│   │   │   ├── raw_games.py                    # Parse Games.csv → ingested format
│   │   │   ├── teams_history.py                # Load NBA teams metadata
│   │   │   ├── parse_league_schedule.py        # Parse league schedule CSV
│   │   │   ├── append_games_results.py         # Merge results into historical data
│   │   │   └── incremental/                    # Agent-based incremental ingestion
│   │   │       ├── agents.py                   # OpenAI agents for data fetching
│   │   │       ├── pipeline.py                 # Incremental ingestion orchestrator
│   │   │       ├── schema.py                   # Pydantic schemas for incremental data
│   │   │       ├── config.py                   # Incremental ingestion configuration
│   │   │       └── io.py                       # I/O helpers for incremental data
│   │   │
│   │   ├── collectors/
│   │   │   ├── upcoming_games.py               # Fetch upcoming games from schedule
│   │   │   ├── upcoming_games_results.py       # Fetch results for played games
│   │   │   └── fetch_game/
│   │   │       ├── get_teams_locations.py      # Fetch team GPS coordinates
│   │   │       ├── make_distances_table.py     # Build team-to-team distance table
│   │   │       └── find_distances.ipynb        # Notebook: distance exploration
│   │   │
│   │   ├── features/
│   │   │   ├── aggregator.py                   # create_features_tables() + merge_features()
│   │   │   ├── winning_percentage.py           # Rolling win % (home/away/total)
│   │   │   ├── point_differential.py           # Rolling point differential
│   │   │   ├── east_vs_west.py                 # Conference win/loss records
│   │   │   ├── rest_days.py                    # Days since last game
│   │   │   ├── distances.py                    # Rolling travel distance
│   │   │   └── last_season_record.py           # Last season's win percentage record
│   │   │
│   │   ├── transformation/
│   │   │   └── add_conference.py               # Add conference column to games
│   │   │
│   │   ├── utils/
│   │   │   └── common.py                       # Shared ETL utility functions
│   │   │
│   │   ├── make_features.py                    # Entry point: build + merge all features
│   │   ├── process_ingested_games.py           # Entry point: ingested → processed
│   │   └── full_pipeline.py                    # High-level pipeline orchestrator
│   │
│   ├── ml/
│   │   ├── README.md                           # ML module documentation
│   │   ├── config/
│   │   │   ├── schema.py                       # ExperimentConfig and Pydantic schemas
│   │   │   └── loader.py                       # Load YAML into config objects
│   │   │
│   │   ├── datasets/
│   │   │   ├── loaders.py                      # Load games_features.csv as DataFrame
│   │   │   └── splitters.py                    # Temporal / random / stratified splits
│   │   │
│   │   ├── features/
│   │   │   ├── engineering.py                  # Delta features, conference features
│   │   │   ├── preprocessing.py                # Scaling, imputation, outlier handling
│   │   │   └── selection.py                    # Boruta-SHAP feature selection
│   │   │
│   │   ├── models/
│   │   │   ├── trainer.py                      # ModelTrainer (train, evaluate, CV)
│   │   │   ├── registry.py                     # ModelRegistry (save/load sklearn models)
│   │   │   └── baseline.py                     # Naive baseline models
│   │   │
│   │   ├── training/                           # Training orchestration
│   │   │   ├── data_prep.py                    # Data preparation helpers
│   │   │   ├── experiment.py                   # Experiment definition and execution
│   │   │   ├── model_factory.py                # Build models from config
│   │   │   └── runners.py                      # High-level training runners
│   │   │
│   │   ├── evaluation/
│   │   │   ├── metrics.py                      # Classification metrics
│   │   │   ├── visualization.py                # Confusion matrix, ROC, feature importance
│   │   │   └── analysis.py                     # Prediction error pattern analysis
│   │   │
│   │   ├── tracking/
│   │   │   ├── mlflow_tracker.py               # MLflowTracker (log params, metrics, model)
│   │   │   ├── delete_experiment.py            # Delete MLflow experiments by name
│   │   │   └── delete_model.py                 # Delete MLflow registered models
│   │   │
│   │   ├── prediction/
│   │   │   └── io.py                           # Load upcoming games DataFrames
│   │   │
│   │   ├── scripts/                            # Entry points (called by Makefile)
│   │   │   ├── train_classifier.py             # Train model from YAML config
│   │   │   ├── predict_classifier.py           # Predict upcoming games from config
│   │   │   ├── predict_upcoming.py             # Wrapper for predict_classifier
│   │   │   ├── run_experiments.py              # Run multiple experiments in batch
│   │   │   └── place_bets.py                   # Place bets via Polymarket API
│   │   │
│   │   └── utils/
│   │       └── validation.py                   # Data validation utilities
│   │
│   └── utils/
│       └── logging_config.py                   # Shared logging setup
│
├── outputs/                                    # Training artifacts (gitignored)
│   ├── all/                                    # Outputs for train_all experiment
│   │   ├── tables/                             # CV result tables
│   │   ├── analysis/                           # Error analysis outputs
│   │   ├── feature_selection/                  # Boruta-SHAP outputs
│   │   └── visualizations/                     # Plots (ROC, confusion matrix, etc.)
│   ├── different/                              # Outputs for train_different experiment
│   │   ├── tables/
│   │   ├── analysis/
│   │   ├── feature_selection/
│   │   └── visualizations/
│   └── same/                                   # Outputs for train_same experiment
│       ├── tables/
│       ├── analysis/
│       ├── feature_selection/
│       └── visualizations/
│
├── .claude/                                    # Claude Code configuration
│   ├── docs/
│   │   └── PROJECT_STRUCTURE.md                # This file
│   ├── experts/
│   │   ├── software-dev.md                     # Software dev expert persona
│   │   └── data-scientist.md                   # Data scientist expert persona
│   ├── hooks/
│   │   └── check-structure-drift.sh            # Stop hook: detects file tree drift
│   ├── skills/
│   │   ├── data-analyst/
│   │   │   ├── SKILL.md                        # Data analyst skill definition
│   │   │   └── references/                     # Quality, outliers, audit, features, leakage
│   │   ├── ml-pipeline/
│   │   │   ├── SKILL.md                        # ML pipeline skill definition
│   │   │   └── references/                     # Config schema
│   │   ├── mlflow/
│   │   │   ├── SKILL.md                        # MLflow skill definition
│   │   │   └── references/                     # Naming and registry conventions
│   │   ├── simplify/
│   │   │   ├── SKILL.md                        # Simplify skill definition
│   │   │   └── references/                     # Pythonic style references
│   │   └── sklearn/
│   │       ├── SKILL.md                        # Sklearn skill definition
│   │       └── references/                     # Pipelines, preprocessing, transformers
│   ├── settings.json                           # Claude Code project settings
│   └── settings.local.json                     # Local overrides (gitignored)
│
├── sandbox/                                    # Notebooks for exploration
├── mlruns/                                     # MLflow run tracking data
├── mlflow.db                                   # MLflow backend database
├── Makefile
├── pyproject.toml
├── .python-version
└── CLAUDE.md
```
