# Packaging decision: one uv project with a slim core

**Decision (2026-10-31):** keep a single uv project and one `uv.lock`, not a workspace or
independent projects. `[project.dependencies]` holds only what the stats fetcher needs. Everything
else sits in the `pipeline` and `gpu` extras.

## What was measured

Import closures come from an AST walk from each CLI entry point, lazy imports included.

| Question | Result |
|---|---|
| Version conflicts | None. Every package resolves in one lock, with Pydantic 2.12.5 throughout. |
| Shared code | `src/config/` and `src/utils/logging_config.py` are used everywhere; `src/etl/utils/common.py` by the fetcher, prediction and monitoring. |
| Fetcher imports | Only `nba_api`, `pandas`, `requests`, and nothing from `src/ml`, `src/betting` or `src/monitoring`. |
| Fetcher environment | 12 packages, against 249 in the lock. |
| Workspace migration cost | 507 import references in 154 files, 40 `python -m src.…` calls, plus `PROJECT_ROOT`, Drive sync and the smoke test. |

The components are less independent than they look:

- `src/betting/game_slugs.py` imports `src.ml.config.loader` and reads `games_features.csv`.
- `fetch_upcoming_games` reads the processed league schedule and team history.
- The real data flow is 1 → 3 (ETL) → 2 → 3 (evaluation), and Polymarket data is only a benchmark.

## Why not the alternatives

- **Independent projects** mainly isolate version conflicts, and there are none. A path dependency
  would not isolate one anyway: it joins the consumer's resolution, so conflicting components have
  to communicate through files or subprocesses.
- **A workspace** would enforce dependency boundaries, but only one deployable needs that, and
  `make check-fetch-env` covers it. A workspace would also break Colab's `pip install`: a virtual
  root has no `[project]` table, and pip ignores `tool.uv.sources`. That is reasoned from how the
  tools work; it was not run.

The trade-off is that extras do not stop a heavy import outside the fetcher's CLIs. Only the
fetcher's boundary is checked.

## What would change this

- **Move to a workspace** if a second deployable appears with a different dependency set, a
  deployed image must not contain the model code, or a package gets published. Extract only `core`
  and that deployable.
- **Split one component out** only if `uv lock` fails on a real conflict, such as an injury scraper
  needing `pydantic<2`. Connect it through files. Try `[tool.uv] conflicts` first if the clash is
  between two extras that never run together.

## Known gaps

- **Colab ignores `uv.lock`.** `pip install .[pipeline,gpu]` resolves fresh, so versions can drift
  from local. A fresh Linux resolve on this date gave pandas 3.1.0, torch 2.15.0 and transformers
  5.22.1, against 2.3.1, 2.9.1 and 5.18.0 in the lock. Packages Colab preinstalls may stay at
  Colab's versions, so this is the possible drift, not what a runtime has. Pinning could shift
  results, so it belongs between experiment rounds.
- **Remote fetching has no data transport.** The fetcher reads and writes local `data/` paths, some
  built by the ETL, and `drive-sync` does not carry `data/`. `daily-cycle` also scores predictions
  between fetching and appending results, so a split needs a defined hand-off.
- **No deployment config exists in this repo** as of this date.
- **stats.nba.com timed out** after 90 seconds from the laptop during this work. Test it from any
  deployment host first.
- **`py-clob-client` was removed** because nothing imported it and no code signs or posts orders.
  Add it back in its own extra when live order placement is built.
