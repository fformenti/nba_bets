.PHONY: lint lint-fix format test \
        build-teams-history build-teams-locations build-distances-table \
        build-polymarket-teams build-game-slug-lookup \
        ingest-raw-games parse-league-schedule process-ingested-games build-features \
        fetch-league-schedule fetch-upcoming-games fetch-game-results append-game-results \
        reconcile-postponed retry-unresolved migrate-incremental \
        predict-upcoming score-predictions bet-polymarket \
        train build-llm-dataset train-llm evaluate-llm \
        delete-experiment delete-model plot-home-win-ratio \
        historical-etl full-rebuild predict-upcoming-games \
        process-results-pipeline daily-cycle

TRAIN_CONFIG      ?= all_models
PREDICTION_CONFIG ?= predict_classifier
LLM_CONFIG        ?= llama31_8b_qlora
LLM_RUN           ?=
# Where finished-game outcomes come from: nba_api | placeholder.
# See src/etl/collectors/results/ — 'placeholder' reads hand-dropped files.
SOURCE            ?= nba_api
# Game date for bet placement, as YYYY-MM-DD.
GAME_DATE         ?=

# ─── Quality ────────────────────────────────────────────────────────────────

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check . --fix

format:
	uv run ruff format .

test:
	uv run pytest

# ─── Reference tables ───────────────────────────────────────────────────────
#
# build-teams-locations and build-distances-table call paid APIs (OpenAI,
# Serper) and write *inputs* to data/raw/reference/, not build artifacts. They
# are deliberately NOT part of full-rebuild: a data rebuild must never depend on
# a network call that can silently return a worse table than the one it
# overwrites. Run them by hand, only when a team relocates or a new city appears.

build-teams-history:
	uv run python -m src.cli.build_teams_history

build-teams-locations:
	uv run python -m src.cli.build_teams_locations

build-distances-table:
	uv run python -m src.cli.build_distances_table $(if $(LIMIT),--limit $(LIMIT),)

build-polymarket-teams:
	uv run python -m src.cli.build_polymarket_teams

# ─── Historical ETL ─────────────────────────────────────────────────────────

ingest-raw-games:
	uv run python -m src.cli.ingest_raw_games

parse-league-schedule:
	uv run python -m src.cli.parse_league_schedule

process-ingested-games:
	uv run python -m src.cli.process_ingested_games

build-features:
	uv run python -m src.cli.build_features --config configs/features.yaml

# ─── Incremental: upcoming games and their results ──────────────────────────

# Re-pull the schedule. This is what makes makeup dates for postponed games
# visible; without it the schedule freezes on the day it was downloaded.
fetch-league-schedule:
	uv run python -m src.cli.fetch_league_schedule

fetch-upcoming-games:
	uv run python -m src.cli.fetch_upcoming_games

fetch-game-results:
	uv run python -m src.cli.fetch_game_results --source $(SOURCE)

append-game-results:
	uv run python -m src.cli.append_game_results

# Release watched postponed games that now have a new date.
reconcile-postponed:
	uv run python -m src.cli.reconcile_postponed

# Put quarantined games back in the queue, once you know why they stalled.
retry-unresolved:
	uv run python -m src.cli.retry_unresolved

# One-time move to the staged incremental layout. See the module docstring.
migrate-incremental:
	uv run python -m src.cli.migrate_incremental_layout

# ─── Prediction and monitoring ──────────────────────────────────────────────

predict-upcoming:
	uv run python -m src.cli.predict_upcoming --config configs/predict/$(PREDICTION_CONFIG).yaml

# How the emitted predictions actually did, once the games were played.
score-predictions:
	uv run python -m src.cli.score_predictions

# ─── Training ───────────────────────────────────────────────────────────────

# Training does not deploy. Pass PROMOTE=1 to also point predict_classifier.yaml
# at the run this produces; otherwise the run URI is just logged for you to
# promote later.
#
# Defaults to the single xgboost model. Sweep all four families — same features,
# same splits, best one registered — with:
#   make train TRAIN_CONFIG=all_models
train:
	uv run python -m src.cli.train_classifier --config configs/train/$(TRAIN_CONFIG).yaml $(if $(PROMOTE),--promote,)

# Build the LLM dataset from the ML models' splits, so both families are scored
# on the same games. Pass ARGS=--push to upload to the Hub.
build-llm-dataset:
	uv run python -m src.cli.build_llm_dataset --config configs/train_llm/$(LLM_CONFIG).yaml $(ARGS)

# LLM fine-tuning (needs `uv sync --extra gpu` on a CUDA box).
# Pass LLM_RUN=<name> to resume an interrupted run from its Hub checkpoint.
train-llm:
	uv run python -m src.cli.train_llm --config configs/train_llm/$(LLM_CONFIG).yaml $(if $(LLM_RUN),--run-name $(LLM_RUN),)

evaluate-llm:
	uv run python -m src.cli.evaluate_llm --config configs/train_llm/$(LLM_CONFIG).yaml --run-name $(LLM_RUN)

# ─── Betting ────────────────────────────────────────────────────────────────

bet-polymarket:
	uv run python -m src.cli.place_bets $(GAME_DATE)

build-game-slug-lookup:
	uv run python -m src.cli.build_game_slug_lookup

# ─── Ops and analysis ───────────────────────────────────────────────────────

delete-experiment:
	uv run python -m src.cli.delete_experiment $(EXPERIMENT)

delete-model:
	uv run python -m src.cli.delete_model $(ARGS)

plot-home-win-ratio:
	uv run python -m src.cli.plot_home_win_ratio

# ─── Composite pipelines ────────────────────────────────────────────────────

# Rebuilds every derived table from raw inputs. Offline and free by design: the
# two API-backed reference builders are not called here, they produce inputs
# (data/raw/reference/) that this target consumes.
full-rebuild:
	$(MAKE) build-teams-history
	$(MAKE) ingest-raw-games
	$(MAKE) append-game-results
	$(MAKE) process-ingested-games
	$(MAKE) build-features

# full-rebuild plus the league schedule, which only changes between seasons.
historical-etl:
	$(MAKE) parse-league-schedule
	$(MAKE) full-rebuild

predict-upcoming-games:
	$(MAKE) fetch-upcoming-games
	$(MAKE) predict-upcoming

# Ingest half only: pull results in and rebuild features.
process-results-pipeline:
	$(MAKE) fetch-game-results
	$(MAKE) score-predictions
	$(MAKE) append-game-results
	$(MAKE) process-ingested-games
	$(MAKE) build-features

# The full loop: predict → play → score → fold into history → predict again off
# the updated history. Run with SOURCE=placeholder to exercise it without a
# live results feed.
daily-cycle:
	$(MAKE) fetch-league-schedule
	$(MAKE) parse-league-schedule
	$(MAKE) reconcile-postponed
	$(MAKE) fetch-upcoming-games
	$(MAKE) predict-upcoming
	$(MAKE) fetch-game-results
	$(MAKE) score-predictions
	$(MAKE) append-game-results
	$(MAKE) process-ingested-games
	$(MAKE) build-features
