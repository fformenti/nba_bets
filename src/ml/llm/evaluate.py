"""Evaluate a fine-tuned LoRA adapter as a sign classifier.

Kept separate from training so a finished adapter can be re-scored (different
split, different sample size, different revision) without another GPU run.
"""

from pathlib import Path
from typing import Any, Optional

from src.config.paths import DEFAULT_TRAIN_LLM_CONFIG_PATH, LLM_EVAL_OUTPUTS_DIR
from src.ml.config.loader import load_llm_training_config
from src.ml.config.schema import LLMTrainingConfig
from src.ml.tracking.mlflow_tracker import MLflowTracker
from src.ml.llm.dataset import load_hf_split
from src.ml.llm.finetune import (
    build_base_model,
    build_hub_model_id,
    build_tokenizer,
    get_hf_token,
)
from src.ml.llm.testers import BatchedClassificationTester

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _load_adapter_model(config: LLMTrainingConfig, hub_model_id: Optional[str], revision=None):
    """Load the tokenizer and the base model with the fine-tuned adapter applied.

    ``hub_model_id=None`` returns the untuned base model, for baselines.
    """
    tokenizer = build_tokenizer(config)
    model = build_base_model(config, tokenizer)
    if hub_model_id is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, hub_model_id, revision=revision, token=get_hf_token())
    model.eval()
    return tokenizer, model


def build_predictor(config: LLMTrainingConfig, hub_model_id: str, revision=None):
    """Return a ``predictor(datapoint) -> str`` backed by the fine-tuned model."""
    import torch

    tokenizer, model = _load_adapter_model(config, hub_model_id, revision=revision)
    max_new_tokens = config.evaluation.max_new_tokens

    def model_predict(item) -> str:
        inputs = tokenizer(item[config.data.prompt_field], return_tensors="pt").to("cuda")
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        prompt_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(output_ids[0, prompt_len:])

    return model_predict


def build_batch_predictor(config: LLMTrainingConfig, hub_model_id: str, revision=None):
    """Return a ``predict_batch(datapoints) -> list[str]`` that decodes them together.

    Prompts are left-padded so every row's prompt ends at the same column: the
    generated tokens then start at one shared offset, and decoding is greedy so
    the answers match :func:`build_predictor` row for row.
    """
    import torch

    tokenizer, model = _load_adapter_model(config, hub_model_id, revision=revision)
    # Decoder-only generation needs left padding; training keeps right padding.
    tokenizer.padding_side = "left"
    max_new_tokens = config.evaluation.max_new_tokens

    def predict_batch(items) -> list[str]:
        prompts = [item[config.data.prompt_field] for item in items]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return tokenizer.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)

    return predict_batch


def run_evaluation(
    config: LLMTrainingConfig,
    run_name: str,
    split: Optional[str] = None,
    size: Optional[int] = None,
    revision: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Score the adapter for *run_name* and write charts to *output_dir*.

    Returns the metrics dict from :class:`BatchedClassificationTester` plus the
    resolved split and hub model id.
    """
    from transformers import set_seed

    split = split or config.evaluation.split
    size = size or config.evaluation.size
    hub_model_id = build_hub_model_id(config, run_name)

    logger.info(f"Evaluating {hub_model_id} on split '{split}' (n={size})")

    data = load_hf_split(config, split)
    predict_batch = build_batch_predictor(config, hub_model_id, revision=revision)

    set_seed(config.evaluation.seed)
    metrics = BatchedClassificationTester(
        predict_batch,
        data,
        title=run_name,
        size=size,
        batch_size=config.evaluation.batch_size,
        output_dir=output_dir,
    ).run()

    logger.info(f"Accuracy: {metrics['accuracy']:.1%} ({metrics['n_correct']}/{metrics['size']})")
    return {**metrics, "split": split, "hub_model_id": hub_model_id}


def run_tracked_evaluation(
    run_name: str,
    config_path: Optional[Path] = None,
    split: Optional[str] = None,
    size: Optional[int] = None,
    revision: Optional[str] = None,
):
    resolved = Path(config_path) if config_path else DEFAULT_TRAIN_LLM_CONFIG_PATH
    if not resolved.exists():
        raise FileNotFoundError(
            f"LLM training config not found: {resolved}. "
            "Pass config_path= or --config with a valid YAML "
            "(e.g. configs/train_llm/qlora/llama31_8b/baseline.yaml)."
        )
    config = load_llm_training_config(resolved)

    output_dir = LLM_EVAL_OUTPUTS_DIR / run_name

    def _evaluate():
        return run_evaluation(
            config=config,
            run_name=run_name,
            split=split,
            size=size,
            revision=revision,
            output_dir=output_dir,
        )

    mlflow_config = config.tracking.mlflow
    if not mlflow_config.enabled:
        result = _evaluate()
    else:
        with MLflowTracker(
            experiment_name=mlflow_config.experiment_name,
            run_name=f"eval-{run_name}",
            tracking_uri=mlflow_config.tracking_uri,
        ) as tracker:
            result = _evaluate()
            tracker.set_tags({"train_run_name": run_name, "stage": "evaluation"})
            tracker.log_params(
                {
                    "hub_model_id": result["hub_model_id"],
                    "eval_split": result["split"],
                    "eval_size": result["size"],
                    "revision": revision or "main",
                }
            )
            tracker.log_metrics(
                {"eval_accuracy": result["accuracy"], "eval_n_correct": result["n_correct"]}
            )
            tracker.log_artifacts(str(output_dir), artifact_path="eval_charts")

    logger.info(f"Charts written to {output_dir}")
    return result
