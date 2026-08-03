# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Manager

Use `uv` for all Python operations. Never use `pip` or `python` directly.

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
For a full file tree with descriptions, see [PROJECT_STRUCTURE.md](./.claude/docs/PROJECT_STRUCTURE.md).
When you create, delete, or move files under `src/` or `configs/`, update `./.claude/docs/PROJECT_STRUCTURE.md` to reflect the change. A Stop hook will remind you if drift is detected.

Only document tracked source files — ignore `sandbox/`, `mlruns/`, `__pycache__/`.
