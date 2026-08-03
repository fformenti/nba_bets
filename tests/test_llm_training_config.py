"""Tests for LLM training config loading."""

from src.config.paths import DEFAULT_TRAIN_LLM_CONFIG_PATH
from src.ml.config.loader import load_llm_training_config


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


def test_classifier_scores_by_sign_agreement():
    """Tester_Classifiers grades a point-diff prediction on which team it picks."""
    from src.ml.training.utils import Tester_Classifiers

    assert Tester_Classifiers.is_correct(guess=3.0, truth=5.0) == 1
    assert Tester_Classifiers.is_correct(guess=-3.0, truth=-5.0) == 1
    assert Tester_Classifiers.is_correct(guess=3.0, truth=-5.0) == 0
    # A zero-margin truth counts as a home loss, so a positive guess is wrong.
    assert Tester_Classifiers.is_correct(guess=3.0, truth=0.0) == 0
