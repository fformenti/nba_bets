# dead_code/

Code removed from `src/` during the 2026-08 refactor because nothing referenced
it. Kept rather than deleted so it is findable without git archaeology.

Nothing here is imported by the project. The directory is excluded from ruff
(`pyproject.toml` → `[tool.ruff] extend-exclude`) and from pytest
(`testpaths = ["tests"]`), so it neither lints nor runs.

**How something qualified as dead:** a whole-repo search over `src/`, `tests/`,
`configs/`, `Makefile` and notebooks found no reference beyond its own
definition and any re-export in a package `__init__`. Baseline commit:
`78b5836`.

## Modules

### `full_pipeline.py`
*Was* `src/etl/full_pipeline.py`.

Superseded by `src/etl/make_features.py`, which does the same five steps and is
what `make make-features` runs. Unreferenced **and stale**: it was never updated
with the game-difficulty-score arguments (`gds_lags`, `gds_location_lags`,
`gds_beta`), so running it would have written a feature table silently missing
those columns.

*To revive:* don't. Take the argument list from `make_features.py` instead.

### `predict_upcoming.py`
*Was* `src/ml/scripts/predict_upcoming.py`.

Broken — `ModuleNotFoundError: No module named 'src.ml.prediction.pipeline'`. A
wrapper left behind when the prediction package was gutted; stale `.pyc` files
for the deleted `pipeline`, `feature_builder` and `config` modules were still
sitting in `src/ml/prediction/__pycache__`.

The refactor gives `src/ml/prediction/pipeline.py` a real implementation, and
the entry point now lives at `src/cli/predict_upcoming.py`. This file is
superseded, not merely dead.

### `incremental/` + `incremental_ingestion.yaml`
*Was* `src/etl/ingestion/incremental/` and `configs/ingestion/incremental_ingestion.yaml`.

An LLM-agent-driven ingestion path: `agents.py` (OpenAI agents that map unknown
CSV column names onto the project schema, with a heuristic fallback),
`schema.py`, `config.py`, `io.py`, `pipeline.py`. Roughly 600 lines.

Zero imports, zero tests, and no Makefile target — `.PHONY` named an
`incremental` target that was never written. The incremental path that actually
runs is `src/etl/collectors/` → `src/etl/ingestion/append_games_results.py`.

*To revive:* move back under `src/etl/ingestion/`, restore the config to
`configs/ingestion/`, and add a `src/cli/` entry point plus a make target.
`build_heuristic_mapping()` is usable on its own if you only want the
non-LLM column-name normalisation.

## Functions

`dead_code.py` holds 12 unreferenced functions, each with a comment naming the
file and line it came from and why it went unused. Imports are omitted — the
file is a record, not a library, and will not execute as-is. Each entry lists
the names it would need.

Removed from `src/`, and from the `__all__` of the relevant `__init__.py`:

| Function | Came from |
|---|---|
| `load_teams_arena` | `src/etl/features/teams_arena.py:97` |
| `NBATeam` | `src/etl/collectors/fetch_game/get_teams_locations.py:41` |
| `validate_config` | `src/ml/config/loader.py:201` |
| `load_historical_features` | `src/ml/prediction/io.py:68` |
| `find_shares_to_buy_fixed_investment` | `src/ml/scripts/place_bets.py:69` |
| `evaluate`, `evaluate_classifier` | `src/ml/training/utils.py:459,463` |
| `train_model`, `evaluate_model` | `src/ml/models/trainer.py:365,404` |
| `save_model` | `src/ml/models/registry.py:143` |
| `stratified_split` | `src/ml/datasets/splitters.py:240` |
| `plot_learning_curves` | `src/ml/evaluation/visualization.py:225` |
| `filter_minimum_games_played` | `src/ml/training/data_prep.py:12` |

One more removal was too small to preserve: an unused `val_pct` local at
`src/ml/utils/validation.py:234`.

## Deliberately *not* moved here

- `src/ml/tracking/delete_experiment.py`, `delete_model.py` — unreferenced by
  other modules, but they are documented `python -m` ops tools. They kept their
  logic and gained CLI entry points and make targets.
- `src/eda/home_win_ratio_by_season.py` — a standalone analysis script with no
  importers by design. It gained a make target.
