"""Build the LLM's HuggingFace dataset from the ML models' train/val/test splits.

The point of this module is parity. The sklearn models and the LLM must be
scored on the same games, or "did the LLM beat the ML models?" has no answer.
So the splits are not recomputed here — they come from
:func:`src.ml.datasets.splits.build_splits`, the same function
``src/ml/training/experiment.py`` uses. Identical gameIds per split then hold by
construction.

Each row becomes ``{game_id, prompt, completion}``: ``prompt`` is the serialized
feature row, ``completion`` the answer ``data.target_mode`` asks for - the
signed point differential, or `` won``/`` lost`` (src/ml/llm/target.py). Either
way it is scored by ``P(home)`` on its first token, which is what makes the
result comparable to the sklearn models' binary home-win prediction.

The column names are load-bearing, not stylistic. trl's SFT collator picks its
code path by inspecting keys: a ``text`` column routes to language modeling and
``completion`` is silently ignored, while ``prompt`` + ``completion`` routes to
the prompt-completion path that ``completion_only_loss`` needs. Renaming either
column turns completion-only training back off without any error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import pandas as pd

from src.config.secrets import HF_TOKEN_ENV_VAR, require_env
from src.ml.config.loader import load_experiment_config
from src.ml.config.schema import ExperimentConfig, LLMTrainingConfig
from src.ml.datasets.splits import Splits, build_splits
from src.ml.llm.serialization import load_prompt_config, serialize_frame
from src.ml.llm.target import SPECS, TargetSpec, check_dataset_matches
from src.utils.logging_config import get_logger

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict

logger = get_logger(__name__)

# HuggingFace split name -> attribute prefix on Splits
SPLIT_NAMES = {"train": "X_train", "validation": "X_val", "test": "X_test"}

# The completion the model learns to produce. Read from ``games_enriched`` — the
# untouched source frame — rather than from the feature frame or metadata:
# margin is an intermediate column deliberately kept out of both, and sourcing
# it here means the LLM needs no change to the shared experiment config.
TARGET_COLUMN = "margin"

# Trailing league margin std (src/etl/features/point_differential.py).
# Margin spread drifted from ~13 to ~15.7 points across the training window, so
# a raw-points target teaches an old era's magnitudes; scaling by this puts
# every season on one spread, and evaluation multiplies the spread back in.
ERA_SCALE_COLUMN = "margin_std"


def era_scale(splits: Splits, index: pd.Index, reference_std: Optional[float]) -> pd.Series:
    """Per-game ``margin_std / reference_std``.

    Dividing a real margin by it gives
    the completion the model learns, multiplying a completion by it gives real
    points back. All ones when ``reference_std`` is None (unscaled target).
    """
    if reference_std is None:
        return pd.Series(1.0, index=index)
    scale = splits.games_enriched.loc[index, ERA_SCALE_COLUMN] / reference_std
    if scale.isna().any() or (scale <= 0).any():
        raise ValueError(
            f"'{ERA_SCALE_COLUMN}' is missing or non-positive for "
            f"{int(scale.isna().sum() + (scale <= 0).sum())} row(s). Rebuild features "
            "with `make build-features`."
        )
    return scale


def build_split_dataset(
    splits: Splits,
    split_name: str,
    serialization_format: str,
    prompt_config: Optional[dict] = None,
    reference_std: Optional[float] = None,
    spec: TargetSpec = SPECS["regression"],
) -> "Dataset":
    """Serialize one split into a HuggingFace ``Dataset``.

    ``spec`` sets the prompt's cue and the completion's form. With
    ``reference_std`` set, a regression completion is the margin rescaled to an
    era whose margin std is ``reference_std``. ``margin`` and ``margin_scale``
    ride along in every mode, so evaluation scores in real points and never
    reads the truth back out of the completion text.
    """
    from datasets import Dataset

    features = getattr(splits, SPLIT_NAMES[split_name])
    metadata = splits.metadata.reindex(features.index)

    if TARGET_COLUMN not in splits.games_enriched.columns:
        raise ValueError(
            f"'{TARGET_COLUMN}' not found in the source feature table, so the LLM "
            "has no completion to learn. Rebuild features with `make build-features`."
        )
    target = splits.games_enriched.loc[features.index, TARGET_COLUMN]

    # games_enriched is the untouched feature table: per_team reads both teams'
    # raw values and the games-played counters from it, not from `features`.
    prompts = serialize_frame(
        features,
        metadata,
        fmt=serialization_format,
        raw=splits.games_enriched,
        prompt_config=prompt_config,
        cue=spec.cue,
    )
    scale = era_scale(splits, features.index, reference_std)
    completions = [spec.completion(m, s) for m, s in zip(target, scale, strict=True)]

    logger.info(f"{split_name}: {len(prompts)} rows ({serialization_format})")
    return Dataset.from_dict(
        {
            "game_id": [int(g) for g in metadata["gameId"]],
            "prompt": prompts,
            "completion": completions,
            "margin": [int(round(v)) for v in target],
            "margin_scale": [float(v) for v in scale],
        }
    )


def build_llm_dataset(
    llm_config: LLMTrainingConfig,
    experiment_config: Optional[ExperimentConfig] = None,
) -> "DatasetDict":
    """Build train/validation/test splits for the LLM from the ML splits.

    Parameters
    ----------
    llm_config : LLMTrainingConfig
        Supplies ``data.source_experiment_config`` and
        ``data.serialization_format``.
    experiment_config : ExperimentConfig, optional
        Pre-loaded experiment config. When omitted it is loaded from
        ``llm_config.data.source_experiment_config``.

    Returns
    -------
    DatasetDict
        Splits whose gameIds match ``build_splits(experiment_config)`` exactly.
    """
    from datasets import DatasetDict

    if experiment_config is None:
        experiment_config = load_experiment_config(llm_config.data.source_experiment_config)

    spec = SPECS[llm_config.data.target_mode]
    if spec is not SPECS["regression"] and llm_config.data.serialization_format != "per_team":
        raise ValueError(
            f"target_mode '{llm_config.data.target_mode}' needs serialization_format "
            "'per_team'; the other formats end in a fixed margin question."
        )
    logger.info(
        f"Building LLM dataset from {llm_config.data.source_experiment_config} "
        f"(format={llm_config.data.serialization_format}, target={llm_config.data.target_mode})"
    )
    splits = build_splits(experiment_config)

    prompt_config = None
    if llm_config.data.serialization_format == "per_team":
        prompt_config = load_prompt_config(llm_config.data.prompt_config)

    dataset = DatasetDict(
        {
            name: build_split_dataset(
                splits,
                name,
                llm_config.data.serialization_format,
                prompt_config,
                reference_std=llm_config.data.target_reference_std,
                spec=spec,
            )
            for name in SPLIT_NAMES
        }
    )

    total = sum(len(d) for d in dataset.values())
    logger.info(f"LLM dataset built: {total} rows across {len(dataset)} splits")
    return dataset


def load_hf_split(
    config: LLMTrainingConfig,
    split: str,
    *,
    max_samples: Optional[int] = None,
    shuffle_seed: Optional[int] = None,
):
    """Load one split of the LLM's HF dataset, capped to ``max_samples``.

    With ``shuffle_seed`` set, the cap takes a seeded random sample rather
    than the first N rows: splits are date-sorted, so the first N would all
    come from one month. Training, in-training eval and inspection all load
    here, so this is where a dataset built in the other target mode is caught.
    """
    from datasets import load_dataset
    from src.ml.llm.finetune import get_hf_token

    dataset = load_dataset(config.data.dataset_name, token=get_hf_token())
    if split not in dataset:
        raise KeyError(
            f"Split '{split}' not in {config.data.dataset_name}. "
            f"Available: {sorted(dataset.keys())}"
        )
    data = dataset[split]
    check_dataset_matches(data[0], SPECS[config.data.target_mode], config.data.prompt_field)
    if max_samples:
        n = min(max_samples, len(data))
        data = (
            data.shuffle(seed=shuffle_seed).select(range(n))
            if shuffle_seed is not None
            else data.select(range(n))
        )
        logger.info(f"Sampled {n} rows from split '{split}' (max_samples).")
    logger.info(f"Loaded {len(data)} rows from split '{split}'.")
    return data


def push_to_hub(dataset: "DatasetDict", dataset_name: str, private: bool = True) -> None:
    """Upload the dataset to the Hugging Face Hub."""
    from huggingface_hub import login

    token = require_env(
        HF_TOKEN_ENV_VAR,
        "A write-scoped token is required to push the dataset to the Hub.",
    )

    login(token, add_to_git_credential=True)
    dataset.push_to_hub(dataset_name, private=private)
    logger.info(f"Pushed dataset to {dataset_name}")


def summarize(dataset: "DatasetDict") -> pd.DataFrame:
    """Row counts and mean prompt length per split, for a quick sanity check."""
    return pd.DataFrame(
        [
            {
                "split": name,
                "rows": len(split),
                "mean_chars": int(pd.Series([len(t) for t in split["prompt"]]).mean()),
            }
            for name, split in dataset.items()
        ]
    )
