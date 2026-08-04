"""Build the LLM's HuggingFace dataset from the ML models' train/val/test splits.

The point of this module is parity. The sklearn models and the LLM must be
scored on the same games, or "did the LLM beat the ML models?" has no answer.
So the splits are not recomputed here — they come from
:func:`src.ml.datasets.splits.build_splits`, the same function
``src/ml/training/experiment.py`` uses. Identical gameIds per split then hold by
construction.

Each row becomes ``{game_id, prompt, completion}``: ``prompt`` is the serialized
feature row, ``completion`` the signed point differential. Grading is by sign
(see :class:`src.ml.llm.testers.ClassificationTester`), which is what makes the
result comparable to the sklearn models' binary home-win prediction.

The column names are load-bearing, not stylistic. trl's SFT collator picks its
code path by inspecting keys: a ``text`` column routes to language modeling and
``completion`` is silently ignored, while ``prompt`` + ``completion`` routes to
the prompt-completion path that ``completion_only_loss`` needs. Renaming either
column turns completion-only training back off without any error.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

from src.config.secrets import HF_TOKEN_ENV_VAR, require_env
from src.ml.config.loader import load_experiment_config
from src.ml.config.schema import ExperimentConfig, LLMTrainingConfig
from src.ml.datasets.splits import Splits, build_splits
from src.ml.llm.serialization import serialize_frame
from src.utils.logging_config import get_logger

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict

logger = get_logger(__name__)

# HuggingFace split name -> attribute prefix on Splits
SPLIT_NAMES = {"train": "X_train", "validation": "X_val", "test": "X_test"}

# The completion the model learns to produce. Read from ``games_enriched`` — the
# untouched source frame — rather than from the feature frame or metadata:
# pts_diff is an intermediate column deliberately kept out of both, and sourcing
# it here means the LLM needs no change to the shared experiment config.
TARGET_COLUMN = "pts_diff"


def _format_completion(point_diff: float) -> str:
    """Signed point differential as an integer, e.g. ``+7`` / ``-3``.

    Deliberately has no leading space and no decimal part: every token in the
    completion is graded, so the only two things worth spending tokens on are
    the sign and the magnitude. The sign matters more of the two — it is what
    :class:`src.ml.llm.testers.ClassificationTester` scores.
    """
    value = int(round(point_diff))
    return f"{'+' if value >= 0 else '-'}{abs(value)}"


def build_split_dataset(
    splits: Splits,
    split_name: str,
    serialization_format: str,
) -> "Dataset":
    """Serialize one split into a HuggingFace ``Dataset``."""
    from datasets import Dataset

    features = getattr(splits, SPLIT_NAMES[split_name])
    metadata = splits.metadata.reindex(features.index)

    if TARGET_COLUMN not in splits.games_enriched.columns:
        raise ValueError(
            f"'{TARGET_COLUMN}' not found in the source feature table, so the LLM "
            "has no completion to learn. Rebuild features with `make build-features`."
        )
    target = splits.games_enriched.loc[features.index, TARGET_COLUMN]

    prompts = serialize_frame(features, metadata, fmt=serialization_format)
    completions = [_format_completion(v) for v in target]

    logger.info(f"{split_name}: {len(prompts)} rows ({serialization_format})")
    return Dataset.from_dict(
        {
            "game_id": [int(g) for g in metadata["gameId"]],
            "prompt": prompts,
            "completion": completions,
        }
    )


def build_llm_dataset(
    llm_config: LLMTrainingConfig,
    experiment_config: Optional[ExperimentConfig] = None,
    holdout_path: Optional[Path] = None,
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
    holdout_path : Path, optional
        Passed through to :func:`build_splits`; defaults to the frozen holdout.

    Returns
    -------
    DatasetDict
        Splits whose gameIds match ``build_splits(experiment_config)`` exactly.
    """
    from datasets import DatasetDict

    if experiment_config is None:
        experiment_config = load_experiment_config(llm_config.data.source_experiment_config)

    logger.info(
        f"Building LLM dataset from {llm_config.data.source_experiment_config} "
        f"(format={llm_config.data.serialization_format})"
    )
    splits = build_splits(experiment_config, holdout_path=holdout_path)
    if not splits.used_fixed_holdout:
        logger.warning(
            "The source experiment fell back to a temporal split — the LLM test "
            "set will drift as new games arrive, and will not match past sklearn "
            "runs. Run `make build-holdout-set` to freeze the holdout."
        )

    dataset = DatasetDict(
        {
            name: build_split_dataset(splits, name, llm_config.data.serialization_format)
            for name in SPLIT_NAMES
        }
    )

    total = sum(len(d) for d in dataset.values())
    logger.info(f"LLM dataset built: {total} rows across {len(dataset)} splits")
    return dataset


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
