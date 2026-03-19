# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

For a full file tree with descriptions, see [PROJECT_STRUCTURE.md](./.claude/docs/PROJECT_STRUCTURE.md).

## Package Manager

Use `uv` for all Python operations. Never use `pip` or `python` directly.

## Common Commands

```bash
# Training
make train                          # Uses EXPERIMENT=train_classifier by default
make train EXPERIMENT=train_classifier # Explicit experiment name

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

Run train/predict modules with any experiment config (see `configs/train/` and `configs/predict/`):
```bash
uv run python -m src.ml.scripts.train_classifier --config <configs/train/your_experiment.yaml>
uv run python -m src.ml.scripts.predict_classifier --config <configs/predict/your_predict.yaml>
```
Use the YAML that matches the model and data split you are testing (e.g. `train_same.yaml`, `train_different.yaml`).

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

## Context Efficiency

Do not re-read files that are already in the conversation context. After reading or editing a file, use the content already available rather than calling the Read tool again.

### PROJECT_STRUCTURE.md Maintenance

When you create, delete, or move files under `src/` or `configs/`, update `.claude/docs/PROJECT_STRUCTURE.md` to reflect the change. A Stop hook will remind you if drift is detected.

Format: use the existing tree-drawing style with `├──`/`└──` connectors and `# description` annotations. Keep descriptions concise (under 60 chars). Only document tracked source files — ignore `data/`, `sandbox/`, `mlruns/`, `__pycache__/`.

### Code Refactor and Code Creation
You are a Python expert very knowledgeable in the field of Data Science and Data Engineering