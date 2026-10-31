"""Evaluation metrics on an inspection frame: probabilities, calibration, margin."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.ml.llm.diagnostics import (
    apply_platt,
    few_shot_prefix,
    fit_platt,
    magnitude_token_ids,
    paired_log_loss_gap,
    read_inspection_csv,
    recalibrate,
    sign_and_margin,
    summarize,
)


def make_frame(n=400, lean=0.15, seed=0) -> pd.DataFrame:
    """Games whose true home-win probability is ``q``; the model reports ``q + lean``."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.2, 0.8, n)
    won = rng.uniform(size=n) < q
    margin = np.where(won, rng.integers(1, 20, n), -rng.integers(1, 20, n))
    p_home = np.clip(q + lean, 0.01, 0.99)
    return pd.DataFrame(
        {
            "game_id": range(n),
            "truth": [f"{m:+d}" for m in margin],
            "raw_output": ["+5" if p > 0.5 else "-5" for p in p_home],
            "parsed": np.where(p_home > 0.5, 5.0, -5.0),
            "p_plus": p_home,
            "p_minus": 1 - p_home,
            "p_sign_mass": 1.0,
            "p_home": p_home,
            "margin": margin.astype(float),
            "margin_scale": 1.0,
            "expected_margin": (p_home - 0.5) * 20,
        }
    )


def test_summary_reports_probability_metrics_and_margin_rmse():
    summary = summarize(make_frame())

    for key in ("log_loss", "brier", "accuracy_p_home", "share_predicted_home", "calibration"):
        assert key in summary
    assert summary["share_predicted_home"] > summary["truth_home_win_rate"]
    assert {"rmse_points_greedy", "rmse_points_expected", "rmse_points_predict_mean"} <= set(summary)
    assert sum(b["n"] for b in summary["calibration"]) == 400


def test_old_csvs_without_margin_still_summarize():
    frame = make_frame().drop(columns=["margin", "margin_scale", "expected_margin"])
    assert "rmse_points_greedy" not in summarize(frame)


def test_saved_csv_keeps_signs_as_text(tmp_path):
    path = tmp_path / "inspection.csv"
    make_frame(n=20).to_csv(path, index=False)

    frame = read_inspection_csv(path)

    assert frame["raw_output"].str[0].isin(["+", "-"]).all()
    assert summarize(frame)["raw_answered_other"] == 0


def test_platt_removes_a_constant_home_lean():
    frame = make_frame(n=4000, lean=0.15)
    calibrated = recalibrate(frame)
    before, after = summarize(frame), summarize(calibrated)
    home_rate = before["truth_home_win_rate"]

    assert after["log_loss"] < before["log_loss"]
    # parsed keeps the greedy answer, so the home share is read off p_home.
    lean_before = (frame["p_home"] > 0.5).mean() - home_rate
    lean_after = (calibrated["p_home"] > 0.5).mean() - home_rate
    assert abs(lean_after) < abs(lean_before)
    # Each half gets its own fit, so the ranking across halves can shift a little.
    assert after["auc_p_home"] == pytest.approx(before["auc_p_home"], abs=0.005)


def test_platt_fit_from_another_split_is_applied_unchanged():
    fit_on, apply_to = make_frame(seed=1), make_frame(seed=2)
    a, b = fit_platt(fit_on["p_home"], fit_on["truth"].str[0] == "+")

    out = recalibrate(apply_to, calibration_frame=fit_on)

    assert np.allclose(out["p_home"], apply_platt(apply_to["p_home"], a, b))
    # One monotone rescaling never reorders games.
    assert summarize(out)["auc_p_home"] == pytest.approx(summarize(apply_to)["auc_p_home"])


def _series(p_home, y_true, game_ids=None):
    game_ids = range(len(p_home)) if game_ids is None else game_ids
    return pd.DataFrame({"game_id": list(game_ids), "p_home": p_home, "y_true": y_true})


def test_paired_gap_of_a_model_with_itself_is_exactly_zero():
    rng = np.random.default_rng(0)
    frame = _series(rng.uniform(0.2, 0.8, 500), rng.integers(0, 2, 500))

    gap = paired_log_loss_gap(frame, frame, n_boot=200)

    assert gap["gap"] == gap["ci_low"] == gap["ci_high"] == 0.0


def test_paired_gap_is_negative_when_the_first_model_is_better():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    informed = _series(np.where(y == 1, 0.65, 0.35) + rng.uniform(-0.1, 0.1, 2000), y)
    coin = _series(np.full(2000, 0.5), y)

    gap = paired_log_loss_gap(informed, coin, n_boot=500)

    assert gap["ci_high"] < 0
    assert gap["ci_low"] < gap["gap"] < gap["ci_high"]


def test_paired_gap_scores_only_the_shared_games():
    a = _series([0.6] * 5, [1] * 5, game_ids=[1, 2, 3, 4, 5])
    b = _series([0.4] * 4, [1] * 4, game_ids=[3, 4, 5, 6])

    assert paired_log_loss_gap(a, b, n_boot=10)["n"] == 3


class FakeTokenizer:
    """'+' and '-' are tokens 1 and 2; magnitude m is token 10 + m."""

    def __call__(self, text, add_special_tokens=False):
        sign, number = text[0], text[1:]
        return {"input_ids": [1 if sign == "+" else 2, 10 + int(number)]}


def test_magnitude_ids_follow_the_tokenizer():
    ids = magnitude_token_ids(FakeTokenizer(), max_magnitude=3)
    assert ids == {0: 10, 1: 11, 2: 12, 3: 13}


def test_magnitude_ids_are_none_for_digit_level_tokenizers():
    class DigitTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1, *(10 + int(d) for d in text[1:])]}

    assert magnitude_token_ids(DigitTokenizer(), max_magnitude=12) is None


class FakeModel:
    """A causal LM whose last hidden state *is* the log-probabilities it wants.

    The output head is the identity, so the logits equal the hidden state:
    P(+) = 0.75; after "+" the magnitude is 4 or 10, equally likely; after "-"
    it is always 4.
    """

    vocab = 20

    def __init__(self):
        import torch

        self.config = SimpleNamespace()
        self.head = torch.nn.Linear(self.vocab, self.vocab, bias=False)
        self.head.weight.data = torch.eye(self.vocab)

    def get_output_embeddings(self):
        return self.head

    def _hidden(self, probabilities):
        import torch

        hidden = torch.full((1, 1, self.vocab), -1e9)
        for token, p in probabilities.items():
            hidden[0, 0, token] = float(np.log(p))
        return hidden

    def model(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is None:
            hidden = self._hidden({1: 0.75, 2: 0.25})
        elif input_ids[0, 0].item() == 1:
            hidden = self._hidden({14: 0.5, 15: 0.5})
        else:
            hidden = self._hidden({14: 1.0})
        return SimpleNamespace(last_hidden_state=hidden, past_key_values={"cache": 1})


def _inputs():
    import torch

    return {"input_ids": torch.ones(1, 3, dtype=torch.long), "attention_mask": torch.ones(1, 3, dtype=torch.long)}


def test_expected_margin_weights_both_signs_and_magnitudes():
    (p_home,), (expected,) = sign_and_margin(FakeModel(), _inputs(), {"+": 1, "-": 2}, {4: 14, 10: 15})

    assert p_home == pytest.approx(0.75)
    assert expected == pytest.approx(0.75 * 7 - 0.25 * 4)


def test_p_home_is_read_off_the_home_answer_whatever_its_text():
    """The home answer is the label dict's first entry, sign or word."""
    (as_signs,), _ = sign_and_margin(FakeModel(), _inputs(), {"+": 1, "-": 2}, None)
    (as_words,), (expected,) = sign_and_margin(FakeModel(), _inputs(), {" won": 1, " lost": 2}, None)

    assert as_words == pytest.approx(as_signs) == pytest.approx(0.75)
    assert np.isnan(expected)


def test_sign_probability_is_resolved_beyond_bf16():
    import torch

    # Two logits 0.004 apart are the same number in bf16; fp32 keeps them apart.
    class CloseCall(FakeModel):
        def model(self, input_ids, past_key_values=None, **kwargs):
            hidden = torch.zeros(1, 1, self.vocab)
            hidden[0, 0, 1], hidden[0, 0, 2] = 2.004, 2.0
            return SimpleNamespace(last_hidden_state=hidden, past_key_values=None)

    assert torch.tensor(2.004, dtype=torch.bfloat16) == torch.tensor(2.0, dtype=torch.bfloat16)
    (p_home,), (expected,) = sign_and_margin(CloseCall(), _inputs(), {"+": 1, "-": 2}, None)

    assert p_home == pytest.approx(1 / (1 + np.exp(-0.004)), rel=1e-4)
    assert p_home > 0.5
    assert np.isnan(expected)


def _tiny_llama():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    # On macOS, torch's OpenMP and the one lightgbm loaded in an earlier test
    # segfault together in a multithreaded forward. One thread is plenty here.
    torch.set_num_threads(1)
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=40,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        pad_token_id=0,
    )
    return LlamaForCausalLM(config).eval()


def _left_padded(lengths, pad_id=0, seed=1):
    import torch

    generator = torch.Generator().manual_seed(seed)
    width = max(lengths)
    input_ids = torch.full((len(lengths), width), pad_id)
    attention_mask = torch.zeros((len(lengths), width), dtype=torch.long)
    for row, n in enumerate(lengths):
        input_ids[row, width - n :] = torch.randint(3, 40, (n,), generator=generator)
        attention_mask[row, width - n :] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_single_pass_scoring_matches_generate_and_two_pass_sign_and_margin():
    import torch

    from src.ml.llm.diagnostics import score_prompts

    model, inputs, max_new = _tiny_llama(), _left_padded([5, 3, 4]), 4
    signs, magnitudes = {"+": 5, "-": 6}, {m: 10 + m for m in range(8)}
    with torch.inference_mode():
        probe = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, eos_token_id=None)
        # Row 1's second token as the EOS, so that row stops early and is padded.
        eos = probe[1, inputs["input_ids"].shape[1] + 1].item()
        reference = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            eos_token_id=eos,
            pad_token_id=0,
            return_dict_in_generate=True,
            output_logits=True,
        )
        two_pass = sign_and_margin(model, inputs, signs, magnitudes)
        step_logits, generated, p_home, expected = score_prompts(
            model, inputs, signs, magnitudes, max_new, eos_ids=[eos], pad_id=0
        )

    expected_ids = reference.sequences[:, inputs["input_ids"].shape[1] :]
    assert (expected_ids == 0).any(), "no row stopped early - the padding path went untested"
    assert torch.equal(generated, expected_ids)
    # Every step's logits, not just the picks: a random model's argmax is too
    # flat to notice, say, decode positions off by one.
    torch.testing.assert_close(step_logits, torch.stack(reference.logits, dim=1).float())
    assert p_home == pytest.approx(two_pass[0])
    assert expected == pytest.approx(two_pass[1])


def test_tokenization_check_starts_the_completion_where_trl_does():
    from src.ml.llm.diagnostics import tokenization_check

    class MergingTokenizer:
        """One token per character, except "=+" merges into one."""

        def __call__(self, text):
            ids, i = [], 0
            while i < len(text):
                if text.startswith("=+", i):
                    ids.append(100)
                    i += 2
                else:
                    ids.append(ord(text[i]))
                    i += 1
            return {"input_ids": ids}

        def decode(self, ids):
            return "".join("=+" if t == 100 else chr(t) for t in ids)

    check = tokenization_check(MergingTokenizer(), "m =", "+7")

    assert check["prefix_matches"] is False
    # trl trains from the common prefix: the merged "=+" and the digit.
    assert check["completion_tokens"] == ["=+", "7"]


def test_few_shot_prefix_is_prompt_then_completion():
    shots = [{"prompt": "A =", "completion": "+3"}, {"prompt": "B =", "completion": "-1"}]
    assert few_shot_prefix(shots) == "A =+3\n\nB =-1\n\n"
