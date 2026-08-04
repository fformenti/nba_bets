# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Manager

Use `uv` for all Python operations. Never use `pip` or `python` directly.

## Secrets

No API key or token ever lives in the repo. Values sit in `~/.secrets/nba_bets.env`
(mode 600), and the committed `.envrc` loads them via direnv on `cd`. Keys shared
with other projects (`WANDB_API_KEY`) live in `~/.secrets/env` instead.

Read them only through `src/config/secrets.py::require_env`, at call time — never
`os.getenv` at module scope, and never build an API client at import. Every module
under `src/` must import without credentials present, because
`tests/test_imports_smoke.py` imports all of them.

On a remote GPU box without direnv, `scp` a `.env` next to the repo; `require_env`
falls back to it non-overriding, so the box's own exports still win.

## Architecture Overview

### Layering rules

- **Libraries never have `__main__`; CLI modules never have logic.** Every entry
  point is a thin argparse shell in `src/cli/`, one per Makefile target.
  Enforced by `tests/test_imports_smoke.py`.
- **One split definition.** `src/ml/datasets/splits.py::build_splits` decides
  train/validation/test. Both the sklearn path and the LLM dataset builder call
  it, so the two model families are always scored on the same gameIds.
- **Two configs, two jobs.** `configs/features.yaml` decides which feature
  *tables* get built (ETL); `configs/train/*.yaml` decides which columns a model
  *consumes* (ML). Using one where the other belongs is a bug.
- **`winner` is the winning teamId, not a 0/1 flag.** Models predict 1 = home
  win. `src/monitoring/scoring.py` converts, in one place.

### Data Flow

```
Historical (make historical-etl):
  data/raw/historical/games/Games.csv
    → [ingest-raw-games]       → data/ingested/
    → [process-ingested-games] → data/processed/regular_season/
    → [build-features]         → data/processed/regular_season/games_features.csv

The daily loop (make daily-cycle):
  [fetch-upcoming-games]   league schedule → data/raw/incremental/upcoming_games/
  [predict-upcoming]       → data/predictions/upcoming_games_predictions.csv
       ⋯ games are played ⋯
  [fetch-game-results]     ResultsSource → upcoming_games_results/
  [score-predictions]      predictions ⋈ outcomes → prediction_scorecard.csv + MLflow
  [append-game-results]    → data/ingested/games_updated_history.csv
  [process-ingested-games]
  [build-features]         history now includes the games just played

  [bet-polymarket]         predictions → Polymarket API
```

Outcome retrieval is pluggable (`src/etl/collectors/results/`). Run the loop
without a live feed via `make daily-cycle SOURCE=placeholder`, after dropping
`{gameId}.json` files into `data/raw/incremental/manual_results/`.

## Context Efficiency

Do not re-read files that are already in the conversation context. After reading or editing a file, use the content already available rather than calling the Read tool again.

### PROJECT_STRUCTURE.md Maintenance
For a full file tree with descriptions, see [PROJECT_STRUCTURE.md](./.claude/docs/PROJECT_STRUCTURE.md).
When you create, delete, or move files under `src/` or `configs/`, update `./.claude/docs/PROJECT_STRUCTURE.md` to reflect the change. A Stop hook will remind you if drift is detected.

Only document tracked source files — ignore `sandbox/`, `mlruns/`, `__pycache__/`.
