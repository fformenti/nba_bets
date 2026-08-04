.PHONY: lint lint-fix format test \
        build-teams-history build-teams-locations build-distances-table \
        build-polymarket-teams build-holdout-set build-game-slug-lookup \
        ingest-raw-games parse-league-schedule process-ingested-games build-features \
        fetch-upcoming-games fetch-game-results append-game-results \
        predict-upcoming score-predictions bet-polymarket \
        train train-all build-llm-dataset train-llm evaluate-llm \
        delete-experiment delete-model plot-home-win-ratio \
        historical-etl full-rebuild predict-upcoming-games \
        process-results-pipeline daily-cycle

TRAIN_CONFIG      ?= train_same
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

# ─── Reference tables (rebuilt rarely) ──────────────────────────────────────

build-teams-history:
	uv run python -m src.cli.build_teams_history

build-teams-locations:
	uv run python -m src.cli.build_teams_locations

build-distances-table:
	uv run python -m src.cli.build_distances_table

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

# Freeze the evaluation boundary. Run once: every model, sklearn and LLM alike,
# is scored against this same set of games.
build-holdout-set:
	uv run python -m src.cli.build_holdout_set

# ─── Incremental: upcoming games and their results ──────────────────────────

fetch-upcoming-games:
	uv run python -m src.cli.fetch_upcoming_games

fetch-game-results:
	uv run python -m src.cli.fetch_game_results --source $(SOURCE)

append-game-results:
	uv run python -m src.cli.append_game_results

# ─── Prediction and monitoring ──────────────────────────────────────────────

predict-upcoming:
	uv run python -m src.cli.predict_upcoming --config configs/predict/$(PREDICTION_CONFIG).yaml

# How the emitted predictions actually did, once the games were played.
score-predictions:
	uv run python -m src.cli.score_predictions

# ─── Training ───────────────────────────────────────────────────────────────

train:
	uv run python -m src.cli.train_classifier --config configs/train/$(TRAIN_CONFIG).yaml

train-all:
	uv run python -m src.cli.run_experiments configs/train/train_all.yaml configs/train/train_same.yaml configs/train/train_different.yaml

# Build the LLM dataset from the ML models' splits, so both families are scored
# on the same games. Add --push to upload to the Hub.
build-llm-dataset:
	uv run python -m src.cli.build_llm_dataset --config configs/train_llm/$(LLM_CONFIG).yaml

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

full-rebuild:
	$(MAKE) build-teams-history
	$(MAKE) build-teams-locations
	$(MAKE) build-distances-table
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
	$(MAKE) fetch-upcoming-games
	$(MAKE) predict-upcoming
	$(MAKE) fetch-game-results
	$(MAKE) score-predictions
	$(MAKE) append-game-results
	$(MAKE) process-ingested-games
	$(MAKE) build-features
