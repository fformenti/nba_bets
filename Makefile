.PHONY: teams-history etl incremental train get-upcoming-games

EXPERIMENT ?= my_experiment

teams-history:
	uv run python -m src.etl.ingestion.teams_history

etl:
	uv run python -m src.etl.full_pipeline

# Incremenatl ETL pipeline
get-upcoming-games:
	uv run python src/etl/ingestion/collectors/upcoming_games.py

# incremental:
# 	uv run python src.etl.ingestion.incremental.pipeline

train:
	uv run python -m src.ml.scripts.train_classifier --config configs/$(EXPERIMENT).yaml

predict-upcoming:
	uv run python -m src.ml.scripts.predict_classifier --config configs/predict_upcoming.yaml
	

