# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

For a full file tree with descriptions, see [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md).

## Package Manager

Use `uv` for all Python operations. Never use `pip` or `python` directly.

## Common Commands

```bash
# Training
make train                          # Uses EXPERIMENT=my_experiment by default
make train EXPERIMENT=my_experiment # Explicit experiment name

# ETL / Data pipelines
make ingest-raw-games               # Parse raw NBA Games.csv into ingested format
make process-league-schedule        # Parse league schedule CSV
make make-features                  # Build all feature tables and merge them
make process-ingested-games         # Process newly ingested games

# Incremental / Upcoming games
make get-upcoming-games             # Fetch upcoming games from schedule
make get-upcoming-games-results     # Fetch results for upcoming games
make append-games-results           # Append results to history
make process-results-pipeline       # Full incremental results pipeline

# Prediction & Betting
make predict-upcoming               # Predict on upcoming games
make bet-polymarket                 # Place bets via Polymarket API

# Reference data
make teams-history                  # Load NBA teams metadata
make teams-locations                # Fetch team location coordinates
make make-distances-table           # Build travel distance table
```

Run individual modules directly:
```bash
uv run python -m src.ml.scripts.train_classifier --config configs/my_experiment.yaml
uv run python -m src.ml.scripts.predict_classifier --config configs/predict_upcoming.yaml
```

## Architecture Overview

### Data Flow

```
Historical:
  data/raw/historical/Games.csv
    → [ingest-raw-games]  → data/ingested/   (parsed + filtered games)
    → [process-ingested-games] → data/processed/ (conference added, feature tables built)
    → [make-features]     → data/processed/games_features.csv

Incremental:
  configs/LeagueSchedule25_26.csv
    → [get-upcoming-games] → data/raw/incremental/upcoming_games/
    → [predict-upcoming]   → data/predictions/upcoming_games_predictions.csv
    → [bet-polymarket]     → Polymarket API

After games played:
  [get-upcoming-games-results] → fetch results
  [append-games-results]       → merge into historical
  [process-results-pipeline]   → rebuild features
```

### Key Directories

See [PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for the full annotated file tree.

- `configs/` — YAML configs for experiments (`ExperimentConfig`) and predictions (`PredictionConfig`)
- `src/config/paths.py` — all data path constants (always import from here, never hardcode paths)
- `src/etl/` — Data pipeline: `ingestion/`, `collectors/`, `features/`, `transformation/`
- `src/ml/` — ML pipeline: `config/schema.py`, `scripts/`, `models/`, `features/`, `tracking/`, `prediction/`
- `sandbox/`- directory for playing around a testing code. Ignore it

### ML Pipeline

**Training** (`src/ml/scripts/train_classifier.py`):
1. Load config (`ExperimentConfig` from YAML)
2. Load `games_features.csv`, filter by date & conference_filter
3. Apply conference features (`apply_conference_features`) based on `conference_filter`
4. Create delta features (home − away for each lag)
5. Temporal split → preprocess (scale + impute) → train sklearn model
6. Log everything to MLflow (`nba_bets_classification` experiment)
7. Register model as `nba_classification_{model_type}_{conference_filter}`

**Prediction** (`src/ml/scripts/predict_classifier.py`):
- Mirrors training feature engineering exactly (same config-driven logic)
- Loads model from MLflow registry via `model_uri` in prediction YAML
- Aligns prediction features to match training feature columns

### Conference Filter

Three modes (set in config): `"all"` | `"same"` | `"different"`
- Determines which games are included and which conference-based features are added
- Must be consistent between training config and prediction config
- MLflow model names include the filter suffix (e.g., `_all`, `_same`, `_different`)

### Configuration Schema

`src/ml/config/schema.py` defines `ExperimentConfig` and `PredictionConfig` (Pydantic models).

Key `ExperimentConfig` fields:
- `data.conference_filter` — filters games and adds appropriate features
- `features.lags`, `features.location_lags`, `features.distances_lags` — temporal lags for features
- `model.type` — `"random_forest"` or `"gradient_boosting"`
- `splitting.method` — `"temporal"` (default), `"random"`, or `"stratified"`

### MLflow

- Tracking: local `mlflow.db` + `mlruns/` directory
- Experiment: `nba_bets_classification`
- Run name format: `{config_name}-{timestamp}`
- Model registry: `nba_classification_{model_type}_{conference_filter}`
- See `docs/MLFLOW_NAMING_GUIDE.md` for naming conventions

### Path Constants

All file paths are in `src/config/paths.py`. Always import paths from there rather than constructing strings manually. Key paths: `REGULAR_SEASON_GAMES_FEATURES_PATH`, `UPCOMING_GAMES_DIR`, `UPCOMING_GAMES_PREDICTIONS_PATH`.

### Claude Code
I will be asking you to create claude code skills, agents, sub agents, mcp servers, slash commands, rules and agent teams. Make sure you optimize all of these components in a way that it uses the minimal amount of space of context window.

### Code Refactor and Code Creation
You are a Python expert very knowledgeable in the field of Data Science and Data Engineering