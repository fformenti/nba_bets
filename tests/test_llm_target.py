"""The two target modes: regression (``+7``) and classification (`` won``).

Everything mode-specific lives in src/ml/llm/target.py; these tests pin that
regression is unchanged and that classification never reaches a code path
that reads its answer as a number.
"""

import contextlib

import numpy as np
import pandas as pd
import pytest

from src.ml.llm.diagnostics import _home_won, label_token_ids, summarize, tokenization_check
from src.ml.llm.target import SPECS, check_dataset_matches, spec_of_completion

REGRESSION, CLASSIFICATION = SPECS["regression"], SPECS["classification"]


def _old_format_completion(point_diff: float) -> str:
    """The regression completion as dataset.py wrote it before target modes existed."""
    value = round(point_diff)
    return f"{'+' if value >= 0 else '-'}{abs(value)}"


@pytest.mark.parametrize("scale", [1.0, 13.0 / 13.0, 15.7 / 13.0, 12.1 / 13.0])
def test_regression_completion_is_byte_identical_to_before(scale):
    for margin in range(-65, 66):
        assert REGRESSION.completion(margin, scale) == _old_format_completion(margin / scale)


def test_classification_completion_is_the_winner():
    assert CLASSIFICATION.completion(1) == " won"
    assert CLASSIFICATION.completion(-1) == " lost"
    # Era scaling is positive, so it never flips the winner.
    assert CLASSIFICATION.completion(3, scale=1.2) == " won"


@pytest.mark.parametrize(
    "spec, raw, parsed",
    [
        (CLASSIFICATION, " won", 1.0),
        (CLASSIFICATION, "lost", -1.0),
        (CLASSIFICATION, " won by a lot", 1.0),
        (CLASSIFICATION, "+7", 0.0),
        (CLASSIFICATION, "", 0.0),
        (REGRESSION, "+7", 7.0),
        (REGRESSION, "-12", -12.0),
        (REGRESSION, " won", 0.0),
    ],
)
def test_parse_reads_each_mode_s_answer(spec, raw, parsed):
    assert spec.parse(raw) == parsed


def test_mode_is_recoverable_from_a_stored_completion():
    assert spec_of_completion(" won") is CLASSIFICATION
    assert spec_of_completion("lost") is CLASSIFICATION
    assert spec_of_completion("+7") is REGRESSION


@pytest.mark.parametrize(
    "row, spec",
    [
        ({"prompt": "...\n\nHome margin =", "completion": "+7"}, CLASSIFICATION),
        ({"prompt": "...\n\nHome team", "completion": " won"}, REGRESSION),
        # A hand-edited row: right cue, wrong completion.
        ({"prompt": "...\n\nHome team", "completion": "+7"}, CLASSIFICATION),
    ],
)
def test_a_dataset_built_in_the_other_mode_is_refused(row, spec):
    with pytest.raises(ValueError, match="target_mode"):
        check_dataset_matches(row, spec)


def test_a_matching_dataset_passes():
    check_dataset_matches({"prompt": "x\n\nHome margin =", "completion": "-3"}, REGRESSION)
    check_dataset_matches({"prompt": "x\n\nHome team", "completion": " lost"}, CLASSIFICATION)


def _llama_tokenizer():
    """The Llama-3 tokenizer, if a cached copy exists; 3.1 and 3.2 share it."""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    for name in ("meta-llama/Llama-3.2-1B", "meta-llama/Meta-Llama-3.1-8B"):
        # Not cached, or no tokenizer backend installed.
        with contextlib.suppress(OSError, ValueError):
            return AutoTokenizer.from_pretrained(name, local_files_only=True)
    pytest.skip("No Llama tokenizer cached locally.")


@pytest.mark.parametrize("mode", ["regression", "classification"])
def test_llama_splits_both_completions_where_the_prompt_ends(mode):
    tokenizer = _llama_tokenizer()
    spec = SPECS[mode]
    prompt = f"Predict the final NBA score margin.\n\n{spec.cue}"

    for margin in (7, -12):
        check = tokenization_check(tokenizer, prompt, spec.completion(margin))
        assert check["prefix_matches"]
        label = spec.home_token if margin > 0 else spec.away_token
        assert check["completion_tokens"][0] == label
        assert len(check["completion_tokens"]) == (2 if spec.has_magnitude else 1)

    ids = label_token_ids(tokenizer, spec)
    assert list(ids) == [spec.home_token, spec.away_token]


def test_a_cue_with_a_trailing_space_breaks_the_boundary():
    """Why the cue is 'Home team' and the space lives in ' won'."""
    tokenizer = _llama_tokenizer()
    check = tokenization_check(tokenizer, "Home team ", "won")
    assert not check["prefix_matches"]


def test_label_ids_refuse_a_word_the_tokenizer_splits():
    class SplitsWords:
        """' won' -> ' w', 'on'; ids are positions in a tiny vocab."""

        vocab = (" w", "on", " lo", "st")

        def __call__(self, text, add_special_tokens=False):
            pieces = {" won": [" w", "on"], " lost": [" lo", "st"]}[text]
            return {"input_ids": [self.vocab.index(p) for p in pieces]}

        def decode(self, ids):
            return "".join(self.vocab[i] for i in ids)

    with pytest.raises(ValueError, match="not a token of its own"):
        label_token_ids(SplitsWords(), CLASSIFICATION)


def _classification_frame(n=200, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    margin = np.where(rng.uniform(size=n) < 0.55, rng.integers(1, 20, n), -rng.integers(1, 20, n))
    p_home = np.clip(0.5 + margin / 60 + rng.normal(0, 0.1, n), 0.01, 0.99)
    return pd.DataFrame(
        {
            "game_id": range(n),
            "truth": [CLASSIFICATION.completion(m) for m in margin],
            "raw_output": np.where(p_home > 0.5, "won", "lost"),
            "parsed": np.where(p_home > 0.5, 1.0, -1.0),
            "p_plus": p_home,
            "p_minus": 1 - p_home,
            "p_sign_mass": 1.0,
            "p_home": p_home,
            "margin": margin.astype(float),
            "margin_scale": 1.0,
            "expected_margin": np.nan,
        }
    )


def test_classification_truth_comes_from_the_margin():
    frame = _classification_frame()
    assert (_home_won(frame) == (frame["margin"] > 0)).all()


def test_classification_summary_has_probabilities_and_no_margin_rmse():
    summary = summarize(_classification_frame())

    assert summary["raw_answered_other"] == 0
    assert summary["raw_answered_home"] == pytest.approx(summary["share_predicted_home"])
    assert summary["accuracy_greedy"] == pytest.approx(summary["accuracy_p_home"])
    for key in ("log_loss", "brier", "auc_p_home", "calibration"):
        assert key in summary
    assert not [key for key in summary if key.startswith("rmse_")]


def test_greedy_tester_grades_won_and_lost():
    from src.ml.llm.testers import BatchedClassificationTester

    data = [
        {"game_id": i, "completion": CLASSIFICATION.completion(m), "margin": m}
        for i, m in enumerate([5, -3, 2, -7])
    ]
    answers = [" won", " won", " lost", " lost"]

    tester = BatchedClassificationTester(
        lambda rows: [answers[row["game_id"]] for row in rows], data, size=4, batch_size=2, spec=CLASSIFICATION
    )
    metrics = tester.run()

    assert metrics == {"accuracy": 0.5, "n_correct": 2, "size": 4}
    # ±1 is not a margin, so none is reported.
    assert tester.predictions("r", "validation")["pred_margin"].isna().all()


@pytest.mark.parametrize(
    "mode, dataset_name",
    [
        # The silent overwrite this guards against: a classification config
        # that kept the regression dataset name it inherited.
        ("classification", "fformenti/nba-bets-compact"),
        ("regression", "fformenti/nba-bets-compact-classification"),
    ],
)
def test_a_dataset_name_from_the_other_mode_is_refused_at_config_load(mode, dataset_name):
    from pydantic import ValidationError

    from src.ml.config.schema import LLMDataConfig

    with pytest.raises(ValidationError, match="-classification"):
        LLMDataConfig(target_mode=mode, dataset_name=dataset_name)


def test_every_shipped_llm_config_names_its_mode_s_dataset():
    from pathlib import Path

    from src.ml.config.loader import load_llm_training_config

    configs = [p for p in Path("configs/train_llm").rglob("*.yaml") if not p.name.startswith("_")]
    loaded = {p.stem: load_llm_training_config(p) for p in configs}

    classification = loaded["compact_classification"].data
    assert classification.target_mode == "classification"
    assert classification.dataset_name == "fformenti/nba-bets-compact-classification"
    assert loaded["compact"].data.dataset_name == "fformenti/nba-bets-compact"
