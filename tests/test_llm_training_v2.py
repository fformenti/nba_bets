"""The v2 training setup: config layering, resume discovery, sampling, best-model rules."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.ml.config.loader import load_llm_training_config
from src.ml.config.schema import LLMTrainerConfig

CONFIGS = Path("configs/train_llm/qlora")


def test_baseline_still_pins_test_run2_settings():
    # test_run2 ran from these values; ../_defaults.yaml has since moved on.
    config = load_llm_training_config(CONFIGS / "llama31_8b" / "baseline.yaml")
    t = config.training

    assert config.quantization.mode == "8bit"
    assert (config.lora.r, config.lora.alpha, config.lora.dropout) == (32, 64, 0.1)
    assert (t.epochs, t.per_device_train_batch_size, t.gradient_accumulation_steps) == (2, 2, 4)
    assert (t.learning_rate, t.warmup_ratio, t.max_grad_norm) == (1e-4, 0.03, 0.3)
    assert t.optimizer == "paged_adamw_32bit"
    assert (t.save_steps, t.eval_steps, t.save_total_limit) == (200, 200, 10)


def test_t4_baseline_differs_from_baseline_only_in_dtype_and_eval():
    baseline = load_llm_training_config(CONFIGS / "llama31_8b" / "baseline.yaml")
    t4 = load_llm_training_config(CONFIGS / "llama31_8b" / "baseline_t4.yaml")

    assert t4.quantization.compute_dtype == "float16"
    assert (t4.training.bf16, t4.training.fp16) == (False, True)
    assert t4.lora == baseline.lora
    assert t4.training.learning_rate == baseline.training.learning_rate
    assert t4.training.epochs == baseline.training.epochs


@pytest.mark.parametrize(
    "model, base_model",
    [
        ("llama31_8b", "meta-llama/Meta-Llama-3.1-8B"),
        ("llama32_1b", "meta-llama/Llama-3.2-1B"),
        ("llama32_3b", "meta-llama/Llama-3.2-3B"),
        ("qwen3_1p7b", "Qwen/Qwen3-1.7B-Base"),
        ("qwen3_4b", "Qwen/Qwen3-4B-Base"),
    ],
)
def test_v2_configs_share_one_training_recipe(model, base_model):
    config = load_llm_training_config(CONFIGS / model / "v2.yaml")
    t = config.training

    assert config.base_model == base_model
    assert config.quantization.mode == "4bit"
    assert (t.epochs, t.per_device_train_batch_size * t.gradient_accumulation_steps) == (1, 32)
    assert (t.learning_rate, t.max_grad_norm) == (2e-4, 1.0)
    assert config.data.max_eval_samples == 1000
    assert config.tracking.wandb.log_model == "false"
    assert config.paths.output_dir.endswith(model)


@pytest.mark.parametrize(
    "variant, changed",
    [
        ("bench_8bit", {"quantization.mode": "8bit"}),
        ("bench_bf16", {"quantization.mode": "none"}),
        ("bench_nogc", {"training.gradient_checkpointing": False}),
        (
            "bench_b2x16",
            {"training.per_device_train_batch_size": 2, "training.gradient_accumulation_steps": 16},
        ),
        (
            "bench_nogc_b2x16",
            {
                "training.gradient_checkpointing": False,
                "training.per_device_train_batch_size": 2,
                "training.gradient_accumulation_steps": 16,
            },
        ),
        (
            "bench_nogc_b4x8",
            {
                "training.gradient_checkpointing": False,
                "training.per_device_train_batch_size": 4,
                "training.gradient_accumulation_steps": 8,
            },
        ),
    ],
)
def test_bench_variants_change_one_thing_each(variant, changed):
    bench = load_llm_training_config(CONFIGS / "llama31_8b" / "bench.yaml").model_dump()
    config = load_llm_training_config(CONFIGS / "llama31_8b" / f"{variant}.yaml").model_dump()

    assert bench["data"]["max_train_samples"] == 1600
    assert bench["training"]["eval_strategy"] == "no"
    assert bench["quantization"]["mode"] == "4bit"
    # The E3 reference, pinned in bench.yaml after v2.yaml moved to E3's winner.
    assert bench["training"]["gradient_checkpointing"] is True
    assert (bench["training"]["per_device_train_batch_size"], bench["training"]["gradient_accumulation_steps"]) == (8, 4)
    for dotted, value in changed.items():
        section, key = dotted.split(".")
        assert config[section][key] == value
        config[section][key] = bench[section][key]
    config.pop("description"), bench.pop("description")
    assert config == bench


def test_seed_repeat_changes_only_the_seed():
    v2 = load_llm_training_config(CONFIGS / "llama32_1b" / "v2.yaml").model_dump()
    repeat = load_llm_training_config(CONFIGS / "llama32_1b" / "seed43.yaml").model_dump()

    assert (v2["training"]["seed"], repeat["training"]["seed"]) == (42, 43)
    repeat["training"]["seed"] = 42
    repeat.pop("description"), v2.pop("description")
    assert repeat == v2


def test_early_stopping_requires_best_model_tracking():
    with pytest.raises(ValidationError, match="load_best_model_at_end"):
        LLMTrainerConfig(early_stopping_patience=3)


def test_best_model_needs_saves_on_eval_steps():
    with pytest.raises(ValidationError, match="multiple of eval_steps"):
        LLMTrainerConfig(
            load_best_model_at_end=True, metric_for_best_model="eval_loss", save_steps=150, eval_steps=100
        )
    LLMTrainerConfig(
        load_best_model_at_end=True, metric_for_best_model="eval_loss", save_steps=200, eval_steps=100
    )


def _hub_with_files(files):
    api = MagicMock()
    api.list_repo_files.return_value = files
    return api


def test_resume_downloads_last_checkpoint(tmp_path):
    from src.ml.llm.finetune import RESUME_DIR, find_resumable_checkpoint

    files = ["adapter_model.safetensors", f"{RESUME_DIR}/optimizer.pt", f"{RESUME_DIR}/trainer_state.json"]
    with (
        patch("huggingface_hub.HfApi", return_value=_hub_with_files(files)),
        patch("huggingface_hub.snapshot_download") as download,
    ):
        checkpoint = find_resumable_checkpoint("fformenti/run", tmp_path, token="t")

    assert checkpoint == tmp_path / RESUME_DIR
    assert download.call_args.kwargs["allow_patterns"] == [f"{RESUME_DIR}/*"]


def test_adapter_only_repo_is_not_resumable(tmp_path):
    # What every_save left on the Hub for test_run2: weights, no optimizer state.
    from src.ml.llm.finetune import find_resumable_checkpoint

    files = ["adapter_config.json", "adapter_model.safetensors", "training_args.bin"]
    with (
        patch("huggingface_hub.HfApi", return_value=_hub_with_files(files)),
        patch("huggingface_hub.snapshot_download") as download,
    ):
        assert find_resumable_checkpoint("fformenti/test_run2", tmp_path, token="t") is None
    download.assert_not_called()


@pytest.mark.parametrize("strategy, expected_seed", [("first", None), ("random", 42)])
def test_train_sample_strategy_picks_the_sampling(strategy, expected_seed):
    from src.ml.llm.finetune import load_train_dataset

    config = load_llm_training_config(CONFIGS / "llama32_1b" / "v2.yaml")
    config.data.max_train_samples = 100
    config.data.train_sample_strategy = strategy
    with patch("src.ml.llm.dataset.load_hf_split") as load:
        load_train_dataset(config)

    assert load.call_args.kwargs == {"max_samples": 100, "shuffle_seed": expected_seed}


def test_eval_sample_ignores_the_training_seed():
    # E7 (training.seed 43) drew a different 1,000-game eval sample from E4's,
    # so the two eval curves could not be compared.
    from src.ml.llm.finetune import load_eval_dataset

    seeds = []
    for name in ("v2.yaml", "seed43.yaml"):
        config = load_llm_training_config(CONFIGS / "llama32_1b" / name)
        with patch("src.ml.llm.dataset.load_hf_split") as load:
            load_eval_dataset(config)
        seeds.append(load.call_args.kwargs["shuffle_seed"])

    assert seeds == [42, 42]
