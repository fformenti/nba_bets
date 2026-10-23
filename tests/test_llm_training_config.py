"""Tests for LLM training config loading."""

from src.config.paths import DEFAULT_TRAIN_LLM_CONFIG_PATH
from src.ml.config.loader import load_llm_training_config


def test_default_llm_config_loads():
    config = load_llm_training_config(DEFAULT_TRAIN_LLM_CONFIG_PATH)

    assert config.base_model == "meta-llama/Meta-Llama-3.1-8B"
    assert config.data.context_length == 131072
    # The notebook silently ran 8-bit despite setting QUANT_4_BIT=True.
    assert config.quantization.mode == "8bit"
    assert config.lora.r == 32
    assert config.tracking.mlflow.enabled is True


def test_8bit_load_silences_only_the_fp16_cast_notice(caplog):
    import logging

    from src.ml.llm.finetune import build_quant_config

    build_quant_config(load_llm_training_config(DEFAULT_TRAIN_LLM_CONFIG_PATH))
    bnb = logging.getLogger("bitsandbytes.autograd._functions")
    with caplog.at_level(logging.WARNING, logger=bnb.name):
        bnb.warning("MatMul8bitLt: inputs will be cast from %s to float16 during quantization", "torch.bfloat16")
        bnb.warning("some other bitsandbytes warning")

    passed = [r.getMessage() for r in caplog.records if r.name == bnb.name]
    assert passed == ["some other bitsandbytes warning"]


def test_run_name_resolution_prefers_cli_then_config():
    from src.ml.llm.finetune import build_hub_model_id, resolve_run_name

    config = load_llm_training_config(DEFAULT_TRAIN_LLM_CONFIG_PATH)

    assert resolve_run_name(config, "explicit") == "explicit"
    assert build_hub_model_id(config, "explicit") == "fformenti/explicit"

    config.run_name = "from-config"
    assert resolve_run_name(config, None) == "from-config"

    # Falls back to a generated, project-prefixed timestamp.
    config.run_name = None
    assert resolve_run_name(config, None).startswith("nba-bets-")


def test_classifier_scores_by_sign_agreement():
    """ClassificationTester grades a point-diff prediction on which team it picks."""
    from src.ml.llm.testers import ClassificationTester

    assert ClassificationTester.is_correct(guess=3.0, truth=5.0) == 1
    assert ClassificationTester.is_correct(guess=-3.0, truth=-5.0) == 1
    assert ClassificationTester.is_correct(guess=3.0, truth=-5.0) == 0
    # A zero-margin truth counts as a home loss, so a positive guess is wrong.
    assert ClassificationTester.is_correct(guess=3.0, truth=0.0) == 0


def test_post_process_keeps_the_sign_of_integer_predictions():
    """The model emits integers; a dropped "-" turns every away pick into a home pick."""
    from src.ml.llm.testers import ClassificationTester

    assert ClassificationTester.post_process("-7") == -7.0
    assert ClassificationTester.post_process("+12") == 12.0
    assert ClassificationTester.post_process("-3<|end_of_text|>") == -3.0
    assert ClassificationTester.post_process(" -2.5 points") == -2.5
    assert ClassificationTester.post_process("no number") == 0


def test_batched_classifier_scores_in_chunks():
    """BatchedClassificationTester calls the predictor once per chunk and keeps row order."""
    from src.ml.llm.testers import BatchedClassificationTester

    data = [
        {"game_id": i, "completion": c} for i, c in enumerate(["+5", "-3", "+2", "-7", "+1"])
    ]
    answers = {0: "+4", 1: "+1", 2: "+9", 3: "-2", 4: "-6"}
    calls = []

    def predict_batch(rows):
        calls.append(len(rows))
        return [answers[row["game_id"]] for row in rows]

    tester = BatchedClassificationTester(predict_batch, data, title="t", size=5, batch_size=2)
    metrics = tester.run()

    assert calls == [2, 2, 1]
    assert metrics == {"accuracy": 3 / 5, "n_correct": 3, "size": 5}
    # It writes nothing: the scored games come back in the shared result schema.
    predictions = tester.predictions("r", "validation")
    assert predictions["game_id"].tolist() == [0, 1, 2, 3, 4]
    assert predictions["model"].unique().tolist() == ["r-greedy"]
    assert (predictions["y_pred"] == predictions["y_true"]).sum() == 3
    assert predictions["p_home"].isna().all()


class _WhitespaceTokenizer:
    """One token per word; BOS (id 0) prepended unless add_special_tokens=False."""

    def __call__(self, texts, add_special_tokens=True):
        bos = [0] if add_special_tokens else []
        return {"input_ids": [bos + [1] * len(t.split()) for t in texts]}


def test_sequence_lengths_count_bos_prompt_completion_and_eos():
    from src.ml.llm.finetune import sequence_lengths

    lengths = sequence_lengths(["a b c", "a"], ["+7", "-3"], _WhitespaceTokenizer())

    # BOS + prompt words + completion + EOS.
    assert lengths == [1 + 3 + 1 + 1, 1 + 1 + 1 + 1]


def test_check_context_length_returns_longest_across_splits():
    from src.ml.llm.finetune import check_context_length

    longest = check_context_length(
        {"train": [10, 40], "test": [55, 5]},
        {"train": ["g1", "g2"], "test": ["g3", "g4"]},
        context_length=100,
    )

    assert longest == 55


def test_check_context_length_raises_naming_the_offending_record():
    import pytest

    from src.ml.llm.finetune import check_context_length

    with pytest.raises(ValueError, match=r"game_id=g3 in split 'test' is 155 tokens"):
        check_context_length(
            {"train": [10, 40], "test": [155, 5]},
            {"train": ["g1", "g2"], "test": ["g3", "g4"]},
            context_length=100,
        )


def test_diagnostics_summary_separates_bias_from_no_signal():
    """A model that ranks perfectly but never crosses 0.5 still gets AUC 1."""
    import pandas as pd

    from src.ml.llm.diagnostics import summarize

    frame = pd.DataFrame(
        {
            "truth": ["+5", "-3", "+8", "-6"],
            "raw_output": ["+2", "+1", "+4", "+1"],
            "parsed": [2.0, 1.0, 4.0, 1.0],
            "p_plus": [0.70, 0.55, 0.80, 0.52],
            "p_minus": [0.30, 0.45, 0.20, 0.48],
            "p_sign_mass": [1.0, 1.0, 1.0, 1.0],
            "p_home": [0.70, 0.55, 0.80, 0.52],
        }
    )

    summary = summarize(frame)

    assert summary["raw_starts_with_minus"] == 0
    assert summary["accuracy_greedy"] == 0.5
    assert summary["auc_p_home"] == 1.0
    assert summary["accuracy_at_median_threshold"] == 1.0


def test_first_completion_token_hits_scores_only_the_first_completion_token():
    # trl's fused LM head returns no logits, only per-position is_top1 /
    # label_mask, both already shifted by one. Row 0: completion of two
    # tokens, sign right. Row 1: sign wrong. Row 2: completion truncated away.
    import torch

    from src.ml.llm.finetune import first_completion_token_hits

    label_mask = torch.tensor(
        [
            [False, True, True],
            [False, True, True],
            [False, False, False],
        ]
    )
    is_top1 = torch.tensor(
        [
            [False, True, False],  # sign right
            [True, False, True],  # sign wrong; a prompt hit and the 2nd token must not count
            [True, True, True],  # no completion - excluded
        ]
    )

    correct, total = first_completion_token_hits(is_top1, label_mask)

    assert correct.item() == 1
    assert total.item() == 2
