.PHONY: teams-history etl incremental train train-all get-upcoming-games

TRAIN_CONFIG ?= train_same
PREDICTION_CONFIG ?= predict_classifier

teams-history:
	uv run python -m src.etl.ingestion.teams_history

teams-locations:
	uv run python -m src.etl.collectors.fetch_game.get_teams_locations

make-distances-table:
	uv run python -m src.etl.collectors.fetch_game.make_distances_table

polymarket-teams-abrev:
	uv run python -m src.data_creation.polymarket_teams_abrev

ingest-raw-games:
	uv run python -m src.etl.ingestion.raw_games

get-upcoming-games-results:
	uv run python src/etl/collectors/upcoming_games_results.py

append-games-results:
	uv run python src/etl/ingestion/append_games_results.py

process-ingested-games:
	uv run python src/etl/process_ingested_games.py

process-league-schedule:
	uv run python -m src.etl.ingestion.parse_league_schedule

get-upcoming-games:
	uv run python src/etl/collectors/upcoming_games.py

predict-upcoming:
	uv run python -m src.ml.scripts.predict_classifier --config configs/predict/$(PREDICTION_CONFIG).yaml

make-features:
	uv run python -m src.etl.make_features --config configs/features.yaml

train:
	uv run python -m src.ml.scripts.train_classifier --config configs/train/$(TRAIN_CONFIG).yaml

train-all:
	uv run python -m src.ml.scripts.run_experiments --config configs/train/*.yaml

bet-polymarket:
	uv run python src/ml/scripts/place_bets.py

historical-etl:
	make teams-history
	make teams-locations
	make make-distances-table
	make ingest-raw-games
	make append-games-results
	make process-league-schedule
	make process-ingested-games
	make make-features

full-rebuild:
	make teams-history
	make teams-locations
	make make-distances-table
	make ingest-raw-games
	make append-games-results
	make process-ingested-games
	make make-features

predict-upcoming-games:
	make get-upcoming-games
	make predict-upcoming

process-results-pipeline:
	make get-upcoming-games-results
	make append-games-results
	make process-ingested-games
	make make-features




