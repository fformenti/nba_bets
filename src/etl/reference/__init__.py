"""Builders for slow-moving reference tables that features join against.

These are not part of the per-run pipeline: they are rebuilt occasionally (a
team relocates, an arena changes, a new season starts) and their output is read
by the feature builders. Kept together because they share that lifecycle —
previously they were scattered across ``ingestion/``, a ``collectors/fetch_game/``
subdirectory that fetched teams rather than games, and a ``data_creation/``
package.

Import directly from submodules, e.g.
``from src.etl.reference.teams_history import load_teams_history_table``.
"""
