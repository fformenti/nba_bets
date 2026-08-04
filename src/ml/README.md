# `src/ml` — modelling

Training, prediction, and the LLM track. Entry points live in `src/cli/`; nothing
in here has a `__main__`.

## Layout

```
src/ml/
├── config/      Pydantic schemas + YAML loading (with includes)
├── datasets/    splits.py is THE split definition; holdout.py freezes it
├── features/    Delta features, conference features, preprocessing, selection
├── models/      ModelTrainer, registry, baselines
├── training/    train_classifier() → train_single_model() → runners
├── llm/         Same experiment, text encoding
├── prediction/  Inference on upcoming games
├── evaluation/  Metrics and plots
├── tracking/    MLflow wrapper + ops tools
└── utils/       Validation, SHAP compatibility
```

## The two invariants

### One split definition

`datasets/splits.py::build_splits(config)` decides which games are train,
validation and test. It is called by `training/experiment.py` **and** by
`llm/dataset.py`. That is deliberate: the sklearn models and the LLM must be
scored on the same games, or comparing them is meaningless.

```python
from src.ml.config.loader import load_experiment_config
from src.ml.datasets.splits import build_splits

splits = build_splits(load_experiment_config("configs/train/train_all.yaml"))
splits.X_train, splits.y_train        # the table the sklearn models consume
splits.game_ids("test")               # the gameIds the LLM must mirror
```

The test split comes from `data/processed/holdout/test_metadata.csv`, frozen
once by `make build-holdout-set`. Without that file `build_splits` falls back to
a temporal split and warns — the test set would then drift as games are added.

`tests/test_llm_split_parity.py` asserts set-equality of gameIds per split
between the two paths.

### Two configs, two jobs

| Config | Decides | Used by |
|---|---|---|
| `configs/features.yaml` | which feature *tables* get built | ETL, and inference-time rebuilds |
| `configs/train/*.yaml` | which columns a model *consumes* | training, feature selection |

An experiment may set `record.lags: []` while leaving a derived group like
`sos_adj_record` enabled — coherent as feature *selection*, incoherent as an ETL
instruction. Passing the experiment config to the ETL builder is what used to
break `predict-upcoming` with `KeyError: 'record_L5'`.
`etl/features/aggregator.py::create_features_tables_from_config` is now the one
place the ETL config is unpacked.

## Training

```bash
make train TRAIN_CONFIG=train_same     # one experiment
make train-all                         # all three conference variants
```

`training/classifier.py::train_classifier` opens an MLflow run, calls
`train_single_model`, and writes the winning model URI back into
`configs/predict/predict_classifier.yaml` so prediction picks it up.

## Prediction

```bash
make predict-upcoming
```

`prediction/pipeline.py` routes each game to the model for its conference
matchup ('same', 'different', 'all'), reading each model's training config back
from its MLflow run so the inference feature set matches the training one.

Output goes to `data/predictions/upcoming_games_predictions.csv` via
`upsert_predictions`, keyed on `(gameId, conference_filter)`. Re-running a slate
replaces rows rather than appending — the accuracy scorecard counts these rows,
so duplicates would corrupt it.

## The LLM track

The LLM is not a separate experiment; it is the same experiment in a different
encoding.

```bash
make build-llm-dataset          # splits → text, summarised (add --push to upload)
make train-llm                  # QLoRA fine-tune (needs a CUDA box)
make evaluate-llm LLM_RUN=...   # score the adapter
```

- `llm/dataset.py` gets its splits from `build_splits`, never its own.
- `llm/serialization.py::serialize_row` is the **only** place a feature row
  becomes text — used for training, validation, testing and inference alike.
  Formats: `markdown` (default), `json`, `prose`.
- `markdown`/`json` follow whatever feature columns the config produces, so a
  new feature reaches the prompt with no code change. `prose` is the original
  hand-written template pinned to a fixed column list; use it only to reproduce
  older runs.
- The serializer strips outcome-bearing columns, so the prompt cannot leak the
  answer even if one reaches the feature frame
  (`tests/test_llm_serialization.py`).

GPU dependencies are an optional extra: `uv sync --extra gpu`. Every torch/peft/
trl import in this package is lazy, so importing it on a laptop is safe.

## Evaluation vs monitoring

- `evaluation/` scores a model against the frozen holdout at training time.
- `src/monitoring/scoring.py` scores the predictions the pipeline actually
  emitted, against games that were actually played (`make score-predictions`).

Note `winner` in the data is the winning **teamId**, not a 0/1 flag, while
models predict 1 = home win. `monitoring/scoring.py` does that conversion in one
place; comparing the two directly yields a plausible-looking but badly wrong
accuracy.
