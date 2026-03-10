.PHONY: teams-history etl incremental train get-upcoming-games

EXPERIMENT ?= my_experiment
PREDICTION_CONFIG ?= predict_upcoming

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

process_ingested_games:
	uv run python src/etl/process_ingested_games.py

process-league-schedule:
	uv run python -m src.etl.ingestion.parse_league_schedule

get-upcoming-games:
	uv run python src/etl/collectors/upcoming_games.py

predict-upcoming:
	uv run python -m src.ml.scripts.predict_classifier --config configs/predict_upcoming.yaml

get-upcoming-games-results:
	uv run python src/etl/collectors/upcoming_games_results.py

append-games-results:
	uv run python src/etl/ingestion/append_games_results.py

process-ingested-games:
	uv run python src/etl/process_ingested_games.py

make-features:
	uv run python src/etl/make_features.py

train:
	uv run python -m src.ml.scripts.train_classifier --config configs/$(EXPERIMENT).yaml

# predict:
# 	uv run python -m src.ml.scripts.predict_classifier --config configs/$(PREDICTION_CONFIG).yaml

bet-polymarket:
	uv run python src/ml/scripts/place_bets.py

historical-etl:
	teams-history
	teams-locations
	make-distances-table
	ingest-raw-games
	process-league-schedule
	make process-ingested-games
	make make-features

predict-upcoming-games:
	make get-upcoming-games
	make predict

process-results-pipeline:
	make get-upcoming-games-results
	make append-games-results
	make process-ingested-games
	make make-features




