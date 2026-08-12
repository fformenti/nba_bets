# `src/ml` — modelling

Training, prediction, and the LLM track. Entry points live in `src/cli/`; nothing
in here has a `__main__`.

## Layout

```
src/ml/
├── config/      Pydantic schemas + YAML loading (with includes)
├── datasets/    splits.py is THE split definition; splitters.py the primitives
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

splits = build_splits(load_experiment_config("configs/train/xgboost.yaml"))
splits.X_train, splits.y_train        # the table the sklearn models consume
splits.game_ids("test")               # the gameIds the LLM must mirror
```

The boundaries are **seasons**, not row proportions: `splitting.test_start_season`
is the first test season and `splitting.val_seasons` how many seasons before it
validate. Both live in `configs/train/_defaults.yaml`, which every train config
includes, so there is one declaration.

Proportions were the old scheme and they failed badly here. Games per season
roughly quintupled since 1950, so a fixed *fraction* of the rows spans far more
calendar time at the old end of the history than the recent end: `val_size: 0.2`
swallowed 2014–2022 whole and no model trained on a game newer than April 2014.
`test_size`/`val_size` survive only as the fallback when the two season keys are
unset, and `build_splits` warns when it has to use them.

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
make train                             # the deployed model (TRAIN_CONFIG=xgboost)
make train PROMOTE=1                   # ...and point prediction at it
make train TRAIN_CONFIG=all_models     # sweep four model families, best one wins
```

Configs under `configs/train/` compose by `_include`: `../features.yaml` supplies
the lags ETL built, `_defaults.yaml` supplies everything else — splits, filters,
per-family hyperparameters and the ML feature set (`delta`/`enabled`). A leaf
config states only its differences, which for `xgboost.yaml` and
`all_models.yaml` is `model.train_models` and nothing more. `llm_features.yaml`
is an `ExperimentConfig` too, not an LLM config: `llm/dataset.py` reads it for
the columns and splits the fine-tune mirrors.

`training/classifier.py::train_classifier` opens an MLflow run, calls
`train_single_model`, and writes the winning model URI back into
`configs/predict/predict_classifier.yaml` so prediction picks it up.

## Prediction

```bash
make predict-upcoming
```

`prediction/pipeline.py` predicts every game with one model, reading that
model's training config back from its MLflow run so the inference feature set
matches the training one. Games used to be routed to a same-conference or
cross-conference model as well; that split scored worse and was removed — see
docs/CONFERENCE_SPLIT.md.

Output goes to `data/predictions/upcoming_games_predictions.csv` via
`upsert_predictions`, keyed on `gameId`. Re-running a slate replaces rows rather
than appending — the accuracy scorecard counts these rows, and the betting path
sizes one order plan per row, so duplicates would corrupt both.

## The LLM track

The LLM is not a separate experiment; it is the same experiment in a different
encoding.

```bash
make build-llm-dataset          # splits → text, summarised (ARGS=--push to upload)
make train-llm                  # QLoRA fine-tune (needs a CUDA box)
make evaluate-llm LLM_RUN=...   # score the adapter
```

- `llm/dataset.py` gets its splits from `build_splits`, never its own.
- `llm/serialization.py::serialize_row` is the **only** place a feature row
  becomes text — used for training, validation, testing and inference alike.
  Formats: `labeled` (default), `json`, `markdown`.
- All three follow whatever feature columns the config produces, so a new
  feature reaches the prompt with no code change. `labeled` additionally
  renders human-readable feature names in grouped sections, which a base
  (non-instruct) model has more to work with.
- The serializer strips outcome-bearing columns, so the prompt cannot leak the
  answer even if one reaches the feature frame
  (`tests/test_llm_serialization.py`).

GPU dependencies are an optional extra: `uv sync --extra gpu`. Every torch/peft/
trl import in this package is lazy, so importing it on a laptop is safe.

## Evaluation vs monitoring

- `evaluation/` scores a model against the test seasons at training time.
- `src/monitoring/scoring.py` scores the predictions the pipeline actually
  emitted, against games that were actually played (`make score-predictions`).

Note `winner` in the data is the winning **teamId**, not a 0/1 flag, while
models predict 1 = home win. `monitoring/scoring.py` does that conversion in one
place; comparing the two directly yields a plausible-looking but badly wrong
accuracy.
