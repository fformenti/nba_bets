# NBA Bets

## Incremental Data Ingestion

Place new raw files in `data/raw/incremental/` (CSV, JSON, or JSONL). The incremental
pipeline will normalize schemas with LLM-assisted agents, append new games to the
historical raw dataset, and refresh processed features.

Run:

```bash
uv run python -m src.etl.ingestion.incremental.pipeline
```

Configuration lives in `configs/incremental_ingestion.yaml`.
