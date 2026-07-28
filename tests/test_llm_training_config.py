"""Tests for LLM training config loading and headless evaluation charts."""

from pathlib import Path

import pytest

from src.config.paths import DEFAULT_TRAIN_LLM_CONFIG_PATH
from src.ml.config.loader import load_llm_training_config
from src.ml.training.utils import Tester_Classifiers


def test_default_llm_config_loads():
    config = load_llm_training_config(DEFAULT_TRAIN_LLM_CONFIG_PATH)

    assert config.base_model == "meta-llama/Meta-Llama-3.1-8B"
    assert config.data.max_sequence_length == 1024
    # The notebook silently ran 8-bit despite setting QUANT_4_BIT=True.
    assert config.quantization.mode == "8bit"
    assert config.lora.r == 32
    assert config.tracking.mlflow.enabled is True


def test_run_name_resolution_prefers_cli_then_config():
    from src.ml.training.llm_finetune import build_hub_model_id, resolve_run_name

    config = load_llm_training_config(DEFAULT_TRAIN_LLM_CONFIG_PATH)

    assert resolve_run_name(config, "explicit") == "explicit"
    assert build_hub_model_id(config, "explicit") == "fformenti/explicit"

    config.run_name = "from-config"
    assert resolve_run_name(config, None) == "from-config"

    # Falls back to a generated, project-prefixed timestamp.
    config.run_name = None
    assert resolve_run_name(config, None).startswith("nba-bets-")


def _fake_data(n=6):
    """Alternating truths; the stub predictor always guesses positive."""
    return [
        {"game_id": f"g{i}", "text": f"prompt {i}", "point_diff": 5 if i % 2 else -5}
        for i in range(n)
    ]


def test_classifier_tester_is_headless_and_returns_metrics(tmp_path: Path):
    data = _fake_data(6)
    tester = Tester_Classifiers(
        lambda item: "3.0", data, title="stub", size=200, output_dir=tmp_path
    )

    # size is clamped to the data length rather than IndexError-ing on 200.
    assert tester.size == 6

    metrics = tester.run()

    # Always guessing positive is right for the 3 positive truths.
    assert metrics == {"accuracy": pytest.approx(0.5), "n_correct": 3, "size": 6}

    for name in ("scatter", "accuracy_trend", "confidence_curve"):
        assert (tmp_path / f"{name}.html").exists()


def test_datapoint_label_prefers_game_id():
    from src.ml.training.utils import _datapoint_label

    assert _datapoint_label({"game_id": 42, "text": "no title marker here"}) == "42"
    assert _datapoint_label({"text": "Title: Lakers vs Heat\nrest"}) == "Lakers vs Heat"
