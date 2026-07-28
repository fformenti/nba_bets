"""Evaluate a fine-tuned LoRA adapter as a sign classifier.

Kept separate from training so a finished adapter can be re-scored (different
split, different sample size, different revision) without another GPU run.
"""

from pathlib import Path
from typing import Any, Optional

from src.ml.config.schema import LLMTrainingConfig
from src.ml.training.llm_finetune import (
    build_base_model,
    build_hub_model_id,
    build_tokenizer,
    get_hf_token,
)
from src.ml.training.utils import Tester_Classifiers
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_eval_dataset(config: LLMTrainingConfig, split: str):
    from datasets import load_dataset

    dataset = load_dataset(config.data.dataset_name, token=get_hf_token())
    if split not in dataset:
        raise KeyError(
            f"Split '{split}' not in {config.data.dataset_name}. "
            f"Available: {sorted(dataset.keys())}"
        )
    return dataset[split]


def build_predictor(config: LLMTrainingConfig, hub_model_id: str, revision=None):
    """Return a ``predictor(datapoint) -> str`` backed by the fine-tuned model."""
    import torch
    from peft import PeftModel

    tokenizer = build_tokenizer(config)
    base_model = build_base_model(config, tokenizer)
    model = PeftModel.from_pretrained(base_model, hub_model_id, revision=revision)
    model.eval()

    max_new_tokens = config.evaluation.max_new_tokens

    def model_predict(item) -> str:
        inputs = tokenizer(item[config.data.text_field], return_tensors="pt").to("cuda")
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        prompt_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(output_ids[0, prompt_len:])

    return model_predict


def run_evaluation(
    config: LLMTrainingConfig,
    run_name: str,
    split: Optional[str] = None,
    size: Optional[int] = None,
    revision: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Score the adapter for *run_name* and write charts to *output_dir*.

    Returns the metrics dict from :class:`Tester_Classifiers` plus the
    resolved split and hub model id.
    """
    from transformers import set_seed

    split = split or config.evaluation.split
    size = size or config.evaluation.size
    hub_model_id = build_hub_model_id(config, run_name)

    logger.info(f"Evaluating {hub_model_id} on split '{split}' (n={size})")

    data = load_eval_dataset(config, split)
    predictor = build_predictor(config, hub_model_id, revision=revision)

    set_seed(config.evaluation.seed)
    metrics = Tester_Classifiers(
        predictor,
        data,
        title=run_name,
        size=size,
        output_dir=output_dir,
    ).run()

    logger.info(
        f"Accuracy: {metrics['accuracy']:.1%} "
        f"({metrics['n_correct']}/{metrics['size']})"
    )
    return {**metrics, "split": split, "hub_model_id": hub_model_id}
