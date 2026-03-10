# Project Structure

```
nba_bets/
├── configs/                                    # YAML configuration files
│   ├── my_experiment.yaml                      # Training config (ExperimentConfig)
│   ├── predict_upcoming.yaml                   # Prediction config (PredictionConfig)
│   ├── incremental_ingestion.yaml              # Incremental ingestion config
│   └── train_classification_example.yaml       # Example training config
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
├── docs/
│   ├── MLFLOW_NAMING_GUIDE.md
│   └── CONFERENCE_FILTER_IMPLEMENTATION.md
│
├── src/
│   ├── config/
│   │   ├── paths.py                            # All data path constants (import from here)
│   │   ├── constants.py                        # Dates, team maps, neutral court labels
│   │   └── aws.py                              # AWS config
│   │
│   ├── data_creation/
│   │   └── polymarket_teams_abrev.py           # Build team abbreviation mapping for Polymarket
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
│   │   │       ├── config.py
│   │   │       └── io.py
│   │   │
│   │   ├── collectors/
│   │   │   ├── upcoming_games.py               # Fetch upcoming games from schedule
│   │   │   ├── upcoming_games_results.py       # Fetch results for played games
│   │   │   └── fetch_game/
│   │   │       ├── get_teams_locations.py      # Fetch team GPS coordinates
│   │   │       └── make_distances_table.py     # Build team-to-team distance table
│   │   │
│   │   ├── features/
│   │   │   ├── aggregator.py                   # create_features_tables() + merge_features()
│   │   │   ├── winning_percentage.py           # Rolling win % (home/away/total)
│   │   │   ├── point_differential.py           # Rolling point differential
│   │   │   ├── east_vs_west.py                 # Conference win/loss records
│   │   │   ├── rest_days.py                    # Days since last game
│   │   │   └── distances.py                    # Rolling travel distance
│   │   │
│   │   ├── transformation/
│   │   │   └── add_conference.py               # Add conference column to games
│   │   │
│   │   ├── utils/
│   │   │   └── common.py
│   │   │
│   │   ├── make_features.py                    # Entry point: build + merge all feature tables
│   │   ├── process_ingested_games.py           # Entry point: transform ingested → processed
│   │   └── full_pipeline.py                    # High-level pipeline orchestrator
│   │
│   ├── ml/
│   │   ├── config/
│   │   │   ├── schema.py                       # ExperimentConfig, PredictionConfig (Pydantic)
│   │   │   └── loader.py                       # Load YAML into config objects
│   │   │
│   │   ├── datasets/
│   │   │   ├── loaders.py                      # Load games_features.csv as DataFrame
│   │   │   └── splitters.py                    # Temporal / random / stratified splits
│   │   │
│   │   ├── features/
│   │   │   ├── engineering.py                  # create_delta_features(), apply_conference_features()
│   │   │   └── preprocessing.py                # Scaling, imputation, outlier handling
│   │   │
│   │   ├── models/
│   │   │   ├── trainer.py                      # ModelTrainer (train, evaluate, cross-validate)
│   │   │   ├── registry.py                     # ModelRegistry (save/load sklearn models)
│   │   │   └── baseline.py                     # Naive baseline models
│   │   │
│   │   ├── evaluation/
│   │   │   ├── metrics.py                      # Classification metrics
│   │   │   └── visualization.py                # Confusion matrix, ROC, feature importance
│   │   │
│   │   ├── tracking/
│   │   │   └── mlflow_tracker.py               # MLflowTracker (log params, metrics, model)
│   │   │
│   │   ├── prediction/
│   │   │   ├── pipeline.py                     # End-to-end prediction pipeline
│   │   │   ├── feature_builder.py              # Build features for upcoming games
│   │   │   ├── config.py                       # Prediction config helpers
│   │   │   └── io.py                           # Load upcoming games DataFrames
│   │   │
│   │   └── scripts/                            # Entry points (called by Makefile)
│   │       ├── train_classifier.py             # Train model from YAML config
│   │       ├── predict_classifier.py           # Predict upcoming games from YAML config
│   │       ├── predict_upcoming.py             # Wrapper for predict_classifier
│   │       └── place_bets.py                   # Place bets via Polymarket API
│   │
│   └── utils/
│       └── logging_config.py                   # Shared logging setup
│
├── sandbox/                                    # Notebooks for exploration
├── outputs/                                    # Training artifacts (plots, etc.)
├── Makefile
├── pyproject.toml
├── CLAUDE.md
└── PROJECT_STRUCTURE.md
```
