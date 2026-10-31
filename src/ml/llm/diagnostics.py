"""Inspect what a fine-tuned adapter actually emits, one game at a time.

:class:`src.ml.llm.testers.ClassificationTester` reduces every answer to a
parsed number, which hides *where* a wrong sign comes from. This module keeps
each stage separate so they can be checked independently:

- **tokenization** - does the first completion token really carry the
  answer (the sign, or `` won``/`` lost``), and does training's joint
  prompt+completion tokenization split where the prompt ends?
- **the model** - the raw decoded text, plus the first-token probability of
  the home and away answer, read straight off the logits.
- **the parser** - the number the target mode extracts from that raw text.

The two target modes (src/ml/llm/target.py) differ only in which token pair
is read and whether a magnitude follows; every metric here is shared.

The probabilities are what separate a model that learned nothing beyond the
home-win base rate (``p_home`` near-constant, AUC ~0.5) from one that ranks
games well but sits on the wrong side of 0.5 (AUC well above 0.5, every
``p_home`` > 0.5) - greedy decoding reports "+" for both.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.ml.llm.target import SPECS, TargetSpec, spec_of_completion
from src.ml.llm.testers import ground_truth
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

TOP_K = 5
# Largest completion magnitude scored by expected_margins. The biggest real
# margin since 1996 is 65 points; era scaling stretches it a little.
MAX_MAGNITUDE = 99
CALIBRATION_BINS = 10


def label_token_ids(tokenizer, spec: TargetSpec = SPECS["regression"]) -> dict[str, int]:
    """Token id of the home and the away answer as the first completion token.

    Home first: :func:`sign_and_margin` reads ``P(home)`` at index 0. Read from
    a tokenized completion (``+7``, `` won``) rather than from the bare label,
    so it matches what training graded. Raises if the tokenizer merged the
    label with what follows, or split a word, because the evaluation rests on
    the first token being the whole answer.
    """
    ids = {}
    for margin, label in ((7, spec.home_token), (-7, spec.away_token)):
        completion = spec.completion(margin)
        first = tokenizer(completion, add_special_tokens=False)["input_ids"][0]
        decoded = tokenizer.decode([first])
        if decoded != label:
            raise ValueError(
                f"First token of {completion!r} decodes to {decoded!r}, not {label!r}: "
                "the answer is not a token of its own under this tokenizer."
            )
        ids[label] = first
    return ids


def magnitude_token_ids(tokenizer, max_magnitude: int = MAX_MAGNITUDE) -> Optional[dict[int, int]]:
    """Token id of each magnitude ``0..max_magnitude`` as the token after a sign.

    ``None`` when some magnitude is not a single token after the sign - Qwen,
    for one, splits numbers into digits - because the expected value then needs
    a multi-step search that :func:`expected_margins` does not do.
    """
    ids = {}
    for magnitude in range(max_magnitude + 1):
        tokens = tokenizer(f"+{magnitude}", add_special_tokens=False)["input_ids"]
        if len(tokens) != 2:
            return None
        ids[magnitude] = tokens[1]
    return ids


def _decoder_and_head(model):
    """The transformer body and the output embedding, beneath any PEFT wrapper.

    LoRA replaces layers inside the body in place, so calling the body directly
    still applies the adapter.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model, base.get_output_embeddings(), base.config


def _restricted_logits(hidden, head, model_config, token_ids):
    """fp32 logits of just ``token_ids`` from the last hidden state.

    The model's own logits are bf16 - about three significant digits - so two
    close sign logits round to the same value: 4.8% of test_run2's validation
    games came out at p_home = 0.5 exactly, and greedy decoding broke each tie
    towards "+". Only a handful of tokens matter here, so computing them in
    fp32 is also cheaper than the full-vocabulary head.
    """
    logits = hidden.float() @ head.weight[token_ids].float().T
    return _softcap(logits, model_config)


def _full_logits(hidden, head, model_config):
    """Next-token logits over the whole vocabulary, in the model's own dtype.

    The same values ``generate`` reports (cast to fp32 afterwards), so greedy
    picks and top-k match it.
    """
    return _softcap(head(hidden).float(), model_config)


def _softcap(logits, model_config):
    cap = getattr(model_config, "final_logit_softcapping", None)
    return cap * (logits / cap).tanh() if cap else logits


def prompt_pass(model, inputs):
    """One forward over left-padded prompts, keeping the cache for what follows.

    Prompts are left-padded, so positions come from the attention mask, as
    ``generate`` computes them.
    """
    decoder, _, _ = _decoder_and_head(model)
    mask = inputs["attention_mask"]
    return decoder(
        input_ids=inputs["input_ids"],
        attention_mask=mask,
        position_ids=(mask.cumsum(-1) - 1).clamp(min=0),
        use_cache=True,
    )


def sign_and_margin(
    model,
    inputs,
    sign_ids: dict[str, int],
    magnitude_ids: Optional[dict[int, int]],
    prompt=None,
) -> tuple[list[float], list[float]]:
    """Per row: fp32 ``P(home)`` and the expected completion ``P(+)·E[m|+] - P(-)·E[m|-]``.

    ``sign_ids`` is :func:`label_token_ids`'s output, home answer first - the
    signs, or `` won``/`` lost`` with ``magnitude_ids=None``.

    Greedy decoding returns the single likeliest magnitude, which for a wide,
    noisy margin is the mode (5 or 7 for test_run2), not the mean - the wrong
    estimate for RMSE. This reads the whole magnitude distribution instead: one
    forward over the prompts, then one step per sign on a copy of its cache.
    Both distributions are renormalized over the tokens that can appear there.
    With ``magnitude_ids=None`` the expected margin is NaN.

    ``prompt`` is a :func:`prompt_pass` output to reuse; its cache is only
    copied, never extended, so it can still seed :func:`greedy_decode`.
    """
    import copy

    import torch

    decoder, head, model_config = _decoder_and_head(model)
    mask = inputs["attention_mask"]
    if prompt is None:
        prompt = prompt_pass(model, inputs)
    sign_probs = torch.softmax(
        _restricted_logits(prompt.last_hidden_state[:, -1], head, model_config, list(sign_ids.values())),
        dim=-1,
    )
    p_home = sign_probs[:, 0]
    if magnitude_ids is None:
        return p_home.tolist(), [float("nan")] * len(p_home)

    values = torch.tensor(list(magnitude_ids), dtype=torch.float32, device=mask.device)
    step_mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
    conditional = {}
    for sign, sign_id in sign_ids.items():
        step = decoder(
            input_ids=torch.full_like(mask[:, :1], sign_id),
            attention_mask=step_mask,
            position_ids=mask.sum(-1, keepdim=True),
            past_key_values=copy.deepcopy(prompt.past_key_values),
            use_cache=True,
        )
        probs = torch.softmax(
            _restricted_logits(step.last_hidden_state[:, -1], head, model_config, list(magnitude_ids.values())),
            dim=-1,
        )
        conditional[sign] = (probs * values).sum(-1)

    home, away = sign_ids
    expected = p_home * conditional[home] - (1 - p_home) * conditional[away]
    return p_home.tolist(), expected.tolist()


def greedy_decode(model, inputs, prompt, max_new_tokens: int, eos_ids: list[int], pad_id: int):
    """Greedy-decode from a :func:`prompt_pass`, as ``generate(do_sample=False)`` would.

    Returns the fp32 logits of every generated position, ``(batch, steps,
    vocab)``, and the generated ids, ``(batch, steps)``. A row that has emitted
    an EOS is filled with ``pad_id`` from then on, and decoding stops once every
    row has. Extends ``prompt``'s cache in place, so run :func:`sign_and_margin`
    first.
    """
    import torch

    decoder, head, model_config = _decoder_and_head(model)
    mask = inputs["attention_mask"]
    logits = _full_logits(prompt.last_hidden_state[:, -1], head, model_config)
    eos = torch.tensor(eos_ids, device=mask.device)
    token = logits.argmax(-1)
    finished = torch.isin(token, eos)
    step_logits, generated = [logits], [token]
    cache = prompt.past_key_values
    lengths = mask.sum(-1, keepdim=True)
    for step in range(1, max_new_tokens):
        if finished.all():
            break
        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
        out = decoder(
            input_ids=token[:, None],
            attention_mask=mask,
            position_ids=lengths + step - 1,
            past_key_values=cache,
            use_cache=True,
        )
        cache = out.past_key_values
        logits = _full_logits(out.last_hidden_state[:, -1], head, model_config)
        token = torch.where(finished, pad_id, logits.argmax(-1))
        finished |= torch.isin(token, eos)
        step_logits.append(logits)
        generated.append(token)
    return torch.stack(step_logits, dim=1), torch.stack(generated, dim=1)


def score_prompts(
    model,
    inputs,
    sign_ids: dict[str, int],
    magnitude_ids: Optional[dict[int, int]],
    max_new_tokens: int,
    eos_ids: list[int],
    pad_id: int,
):
    """Everything inspection reads off a batch, from a single pass over the prompts.

    Returns ``(step_logits, generated, p_home, expected_margin)``: the first
    two as :func:`greedy_decode` gives them, the last two as
    :func:`sign_and_margin` does. Running ``generate`` and then
    :func:`sign_and_margin` costs two passes over ~425-token prompts; here the
    sign steps and the greedy steps both start from one prompt cache.
    """
    prompt = prompt_pass(model, inputs)
    p_home, expected = sign_and_margin(model, inputs, sign_ids, magnitude_ids, prompt=prompt)
    step_logits, generated = greedy_decode(model, inputs, prompt, max_new_tokens, eos_ids, pad_id)
    return step_logits, generated, p_home, expected


def _eos_ids(model) -> list[int]:
    eos = model.generation_config.eos_token_id
    return [eos] if isinstance(eos, int) else list(eos or [])


def few_shot_prefix(examples) -> str:
    """Solved examples to put before each prompt, for an untuned base model.

    Each is the prompt immediately followed by its completion - the exact
    format fine-tuning teaches - then a blank line.
    """
    return "".join(f"{ex['prompt']}{ex['completion']}\n\n" for ex in examples)


def tokenization_check(tokenizer, prompt: str, completion: str) -> dict[str, Any]:
    """Replicate trl's prompt-completion tokenization for one row.

    trl tokenizes ``prompt`` and ``prompt + completion`` separately and trains
    on everything past their common prefix. If the joint tokenization merges
    across the boundary, that prefix is shorter than the prompt: the end of the
    prompt is dropped from training (trl warns once) and the first completion
    token trained on is a merged token, not the sign.
    """
    prompt_ids = tokenizer(prompt)["input_ids"]
    joint_ids = tokenizer(prompt + completion)["input_ids"]
    common = 0
    while common < min(len(prompt_ids), len(joint_ids)) and prompt_ids[common] == joint_ids[common]:
        common += 1
    completion_ids = joint_ids[common:]
    return {
        "prefix_matches": joint_ids[: len(prompt_ids)] == prompt_ids,
        "prompt_tail_tokens": [tokenizer.decode([t]) for t in prompt_ids[-5:]],
        "completion_tokens": [tokenizer.decode([t]) for t in completion_ids],
        "n_prompt_tokens": len(prompt_ids),
    }


def predict_with_first_token_probs(
    model,
    tokenizer,
    rows: list[dict],
    prompt_field: str,
    max_new_tokens: int,
    magnitude_ids: Optional[dict[int, int]] = None,
    spec: Optional[TargetSpec] = None,
) -> list[dict[str, Any]]:
    """Greedy-decode a batch, keeping the first step's answer probabilities.

    ``tokenizer`` must pad on the left, so every prompt ends at the same
    column and the first generated position is shared by all rows.

    ``p_plus``/``p_minus`` are the home and away answer (the signs, or
    `` won``/`` lost``), read off the model-dtype first-step logits, so their
    sum shows whether the model opens with an answer at all; ``p_home`` and
    ``expected_margin`` come from :func:`sign_and_margin` in fp32. ``spec``
    defaults to the mode the rows' completions were built in; a mode without
    a magnitude ignores ``magnitude_ids``.
    """
    import torch

    spec = spec or spec_of_completion(rows[0]["completion"])
    sign_ids = label_token_ids(tokenizer, spec)
    home, away = sign_ids
    if not spec.has_magnitude:
        magnitude_ids = None
    prompts = [row[prompt_field] for row in rows]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        step_logits, generated, p_home, expected = score_prompts(
            model,
            inputs,
            sign_ids,
            magnitude_ids,
            max_new_tokens,
            eos_ids=_eos_ids(model),
            pad_id=tokenizer.pad_token_id,
        )

    first_probs = torch.softmax(step_logits[:, 0], dim=-1)
    top_p, top_ids = first_probs.topk(TOP_K, dim=-1)

    results = []
    for i, row in enumerate(rows):
        p_plus = first_probs[i, sign_ids[home]].item()
        p_minus = first_probs[i, sign_ids[away]].item()
        raw = tokenizer.decode(generated[i], skip_special_tokens=True)
        results.append(
            {
                "game_id": row["game_id"],
                "truth": row["completion"],
                # Real points and the factor taking completion units back to
                # them (1.0 for a dataset built without era scaling).
                "margin": row.get("margin", float("nan")),
                "margin_scale": row.get("margin_scale", 1.0),
                "expected_margin": expected[i],
                "raw_output": raw,
                "raw_output_tokens": "|".join(tokenizer.decode([t]) for t in generated[i]),
                "parsed": spec.parse(raw),
                "p_plus": p_plus,
                "p_minus": p_minus,
                # Mass on the two answers: near 1 means the model always opens
                # with one, as trained; low means it opens with something else.
                "p_sign_mass": p_plus + p_minus,
                # Home-win probability, renormalised over the two answers, in fp32.
                "p_home": p_home[i],
                "top_first_tokens": ", ".join(
                    f"{tokenizer.decode([t])!r}:{p:.3f}"
                    for t, p in zip(top_ids[i].tolist(), top_p[i].tolist(), strict=True)
                ),
            }
        )
    return results


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    """Aggregate the per-game frame into the numbers that locate the fault.

    Needs no GPU, so it can be re-run on a saved CSV. Margin RMSE is reported
    only for CSVs that carry ``margin`` (written since expected_margin existed).
    """
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    home_won = _home_won(frame)
    spec = spec_of_completion(frame["truth"].iloc[0])
    raw = frame["raw_output"].astype(str).str.lstrip()
    answered_home = raw.str.startswith(spec.home_token.strip())
    answered_away = raw.str.startswith(spec.away_token.strip())
    p_home = frame["p_home"].clip(1e-6, 1 - 1e-6)

    summary: dict[str, Any] = {
        "n": len(frame),
        "truth_home_win_rate": home_won.mean(),
        # Whether the greedy answer opens with the home or the away answer of
        # the mode the dataset was built in, or with neither.
        "raw_answered_home": answered_home.mean(),
        "raw_answered_away": answered_away.mean(),
        "raw_answered_other": (~answered_home & ~answered_away).mean(),
        "parsed_negative_rate": (frame["parsed"] < 0).mean(),
        # How often the model picks home, against truth_home_win_rate: the gap
        # is the home lean an era-shifted training prior leaves behind. Read
        # off p_home, not the greedy answer, so it reflects recalibration.
        "share_predicted_home": (p_home > 0.5).mean(),
        "mean_p_sign_mass": frame["p_sign_mass"].mean(),
        "p_home_quantiles": frame["p_home"].quantile([0.0, 0.1, 0.5, 0.9, 1.0]).round(3).to_dict(),
        "accuracy_greedy": ((frame["parsed"] > 0) == home_won).mean(),
        "accuracy_p_home": ((p_home > 0.5) == home_won).mean(),
        "log_loss": log_loss(home_won, p_home, labels=[False, True]),
        "brier": brier_score_loss(home_won, p_home),
        "calibration": calibration_table(p_home, home_won),
    }
    if home_won.nunique() == 2:
        # Ranking quality, independent of where the 0.5 threshold sits.
        summary["auc_p_home"] = roc_auc_score(home_won, frame["p_home"])
        # Accuracy if the threshold were moved to the median p_home, i.e. if
        # the model's bias towards "+" were removed and only its ranking kept.
        summary["accuracy_at_median_threshold"] = (
            (frame["p_home"] > frame["p_home"].median()) == home_won
        ).mean()
    # A won/lost answer parses to ±1, which is no margin to take an RMSE of.
    if spec.has_magnitude and "margin" in frame and frame["margin"].notna().all():
        scale = frame["margin_scale"]
        summary["rmse_points_greedy"] = _rmse(frame["parsed"] * scale - frame["margin"])
        summary["rmse_points_predict_mean"] = _rmse(frame["margin"] - frame["margin"].mean())
        if frame["expected_margin"].notna().all():
            summary["rmse_points_expected"] = _rmse(
                frame["expected_margin"] * scale - frame["margin"]
            )
    return summary


def read_inspection_csv(path) -> pd.DataFrame:
    """Load a saved inspection frame with its text columns kept as text.

    Left to infer, pandas reads a "+5" cell as the integer 5, and the summary's
    raw_answered_* rates would see no signs at all.
    """
    return pd.read_csv(path, dtype={"truth": str, "raw_output": str, "raw_output_tokens": str})


def _home_won(frame: pd.DataFrame) -> pd.Series:
    """Truth from the real ``margin``; only CSVs older than that column parse ``truth``.

    A won/lost completion carries no number, so it must never be parsed here.
    """
    if "margin" in frame and frame["margin"].notna().all():
        return frame["margin"] > 0
    return frame["truth"].map(lambda c: ground_truth({"completion": c})) > 0


def _rmse(errors: pd.Series) -> float:
    return float(np.sqrt((errors**2).mean()))


def calibration_table(p_home: pd.Series, home_won: pd.Series) -> list[dict[str, Any]]:
    """Mean predicted vs actual home-win rate in equal-width probability bins.

    A calibrated model has ``mean_p`` close to ``win_rate`` in every bin;
    test_run2 sat ~5 points above it from 0.6 to 0.8.
    """
    edges = np.linspace(0, 1, CALIBRATION_BINS + 1)
    bins = pd.cut(p_home, edges, include_lowest=True)
    table = (
        pd.DataFrame({"p": p_home, "won": home_won.astype(float), "bin": bins})
        .groupby("bin", observed=True)
        .agg(n=("won", "size"), mean_p=("p", "mean"), win_rate=("won", "mean"))
    )
    return [
        {"bin": str(interval), "n": int(row.n), "mean_p": round(row.mean_p, 3), "win_rate": round(row.win_rate, 3)}
        for interval, row in table.iterrows()
    ]


def fit_platt(p_home: pd.Series, home_won: pd.Series) -> tuple[float, float]:
    """Platt scaling: ``(a, b)`` such that ``sigmoid(a·logit(p) + b)`` is calibrated.

    Two parameters fitted on held-out games. ``b`` absorbs a constant home lean
    (a training prior from an era with more home wins); ``a`` fixes over- or
    under-confidence. Neither changes the ranking, so AUC is untouched.
    """
    from sklearn.linear_model import LogisticRegression

    logit = _logit(p_home).to_numpy().reshape(-1, 1)
    fitted = LogisticRegression(C=1e6).fit(logit, home_won.astype(int))
    return float(fitted.coef_[0, 0]), float(fitted.intercept_[0])


def apply_platt(p_home: pd.Series, a: float, b: float) -> pd.Series:
    return 1 / (1 + np.exp(-(a * _logit(p_home) + b)))


def _logit(p: pd.Series) -> pd.Series:
    p = p.clip(1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def recalibrate(
    frame: pd.DataFrame, calibration_frame: Optional[pd.DataFrame] = None, seed: int = 42
) -> pd.DataFrame:
    """Return ``frame`` with ``p_home`` Platt-recalibrated.

    With ``calibration_frame`` (e.g. the validation inspection), the fit comes
    from it and is applied to ``frame`` (e.g. test). Without one, ``frame`` is
    cross-fitted in two seeded halves, so no game is scored by a fit it was in
    (the two fits differ slightly, so AUC can move a little too). ``parsed``
    keeps the greedy answer; read ``accuracy_p_home`` for the recalibrated
    accuracy.
    """
    out = frame.copy()
    if calibration_frame is not None:
        a, b = fit_platt(calibration_frame["p_home"], _home_won(calibration_frame))
        out["p_home"] = apply_platt(frame["p_home"], a, b)
        return out

    half = np.random.default_rng(seed).permutation(len(frame)) % 2 == 0
    for fit_mask in (half, ~half):
        fit_rows = frame[fit_mask]
        a, b = fit_platt(fit_rows["p_home"], _home_won(fit_rows))
        out.loc[~fit_mask, "p_home"] = apply_platt(frame.loc[~fit_mask, "p_home"], a, b)
    return out


def paired_log_loss_gap(
    a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 5000, seed: int = 0
) -> dict[str, float]:
    """Log-loss of ``a`` minus ``b`` on the games both scored, with a bootstrap interval.

    Each frame holds one calibrated series (``game_id``, ``p_home``,
    ``y_true``), e.g. a run's ``platt`` rows from ``read_predictions``. Pairing
    on ``game_id`` cancels the game-to-game difficulty both models share, so the
    interval is several times tighter than either model's own. Negative means
    ``a`` is better. ``boot_sd`` is the SD of the resampled gap, the "paired SD"
    of docs/findings/llm/2026-10-21-seed-noise.md; ``game_sd`` is the per-game SD,
    which sets how the interval shrinks with more games (as 1/√n).
    """
    joined = (
        a[["game_id", "p_home", "y_true"]]
        .merge(b[["game_id", "p_home"]], on="game_id", suffixes=("_a", "_b"))
        .sort_values("game_id")
    )
    y = joined["y_true"].to_numpy()

    def per_game(p: pd.Series) -> np.ndarray:
        p = p.clip(1e-6, 1 - 1e-6).to_numpy()
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    diff = per_game(joined["p_home_a"]) - per_game(joined["p_home_b"])
    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {
        "n": len(diff),
        "gap": float(diff.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "boot_sd": float(boot.std()),
        "game_sd": float(diff.std()),
    }


def inspect_adapter(
    config,
    run_name: str,
    split: str,
    size: int,
    batch_size: int,
    revision=None,
    base_only: bool = False,
    few_shot: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Run the adapter for *run_name* over *size* rows of *split*.

    ``base_only`` scores the untuned base model instead (``run_name`` then only
    names the output folder); ``few_shot`` puts that many seeded training
    examples, solved, in front of every prompt - the baselines a fine-tune has
    to beat.

    Returns the per-game frame, its :func:`summarize` summary and a
    :func:`tokenization_check` of the first row.
    """
    from tqdm.auto import tqdm

    from src.ml.llm.dataset import load_hf_split
    from src.ml.llm.evaluate import _load_adapter_model
    from src.ml.llm.finetune import build_hub_model_id

    field = config.data.prompt_field
    data = load_hf_split(config, split)
    size = min(size, len(data))
    if few_shot:
        shots = load_hf_split(
            config, config.data.train_split, max_samples=few_shot, shuffle_seed=config.evaluation.seed
        )
        prefix = few_shot_prefix(shots)
        data = data.map(lambda row: {field: prefix + row[field]})

    hub_model_id = None if base_only else build_hub_model_id(config, run_name)
    tokenizer, model = _load_adapter_model(config, hub_model_id, revision=revision)

    spec = SPECS[config.data.target_mode]
    first = data[0]
    tok_check = tokenization_check(tokenizer, first[field], first["completion"])
    magnitude_ids = magnitude_token_ids(tokenizer) if spec.has_magnitude else None
    if spec.has_magnitude and magnitude_ids is None:
        logger.warning(
            "Magnitudes are not single tokens under this tokenizer; "
            "expected_margin (and its RMSE) is skipped."
        )

    tokenizer.padding_side = "left"
    records = []
    for start in tqdm(range(0, size, batch_size)):
        rows = [data[i] for i in range(start, min(start + batch_size, size))]
        records.extend(
            predict_with_first_token_probs(
                model,
                tokenizer,
                rows,
                prompt_field=field,
                max_new_tokens=config.evaluation.max_new_tokens,
                magnitude_ids=magnitude_ids,
                spec=spec,
            )
        )

    frame = pd.DataFrame(records)
    return frame, summarize(frame), tok_check
