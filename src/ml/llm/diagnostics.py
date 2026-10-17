"""Inspect what a fine-tuned adapter actually emits, one game at a time.

:class:`src.ml.llm.testers.ClassificationTester` reduces every answer to a
parsed number, which hides *where* a wrong sign comes from. This module keeps
each stage separate so they can be checked independently:

- **tokenization** - does the first completion token really carry the sign,
  and does training's joint prompt+completion tokenization split where the
  prompt ends?
- **the model** - the raw decoded text, plus the first-token probability of
  ``+`` and ``-``, read straight off the logits.
- **the parser** - the number the tester extracts from that raw text.

The probabilities are what separate a model that learned nothing beyond the
home-win base rate (``p_home`` near-constant, AUC ~0.5) from one that ranks
games well but sits on the wrong side of 0.5 (AUC well above 0.5, every
``p_home`` > 0.5) - greedy decoding reports "+" for both.
"""

from typing import Any

import pandas as pd

from src.ml.llm.testers import ClassificationTester, ground_truth

SIGNS = ("+", "-")
TOP_K = 5


def sign_token_ids(tokenizer) -> dict[str, int]:
    """Token id of each sign as the first completion token.

    Read from a tokenized completion rather than from the bare character, so
    it matches what training graded even if a tokenizer merged the sign with
    the digits (in which case this raises, because the first-token-is-the-sign
    assumption the whole evaluation rests on would be false).
    """
    ids = {}
    for sign in SIGNS:
        first = tokenizer(f"{sign}7", add_special_tokens=False)["input_ids"][0]
        decoded = tokenizer.decode([first])
        if decoded != sign:
            raise ValueError(
                f"First token of '{sign}7' decodes to {decoded!r}, not {sign!r}: "
                "the sign is not a token of its own under this tokenizer."
            )
        ids[sign] = first
    return ids


def tokenization_check(tokenizer, prompt: str, completion: str) -> dict[str, Any]:
    """Replicate trl's prompt-completion tokenization for one row.

    trl 0.22 tokenizes ``prompt`` and ``prompt + completion`` separately and
    marks everything past ``len(prompt_ids)`` as completion. If the joint
    tokenization merges across the boundary, the completion mask is shifted
    and the "first completion token" trained on is not the sign.
    """
    prompt_ids = tokenizer(prompt)["input_ids"]
    joint_ids = tokenizer(prompt + completion)["input_ids"]
    completion_ids = joint_ids[len(prompt_ids) :]
    return {
        "prefix_matches": joint_ids[: len(prompt_ids)] == prompt_ids,
        "prompt_tail_tokens": [tokenizer.decode([t]) for t in prompt_ids[-5:]],
        "completion_tokens": [tokenizer.decode([t]) for t in completion_ids],
        "n_prompt_tokens": len(prompt_ids),
    }


def predict_with_first_token_probs(
    model, tokenizer, rows: list[dict], prompt_field: str, max_new_tokens: int
) -> list[dict[str, Any]]:
    """Greedy-decode a batch, keeping the first step's sign probabilities.

    ``tokenizer`` must pad on the left, so every prompt ends at the same
    column and the first generated position is shared by all rows.
    """
    import torch

    sign_ids = sign_token_ids(tokenizer)
    prompts = [row[prompt_field] for row in rows]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_logits=True,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated = out.sequences[:, prompt_len:]
    # Raw logits of the first generated position, before any logits processor.
    first_probs = torch.softmax(out.logits[0].float(), dim=-1)
    top_p, top_ids = first_probs.topk(TOP_K, dim=-1)

    results = []
    for i, row in enumerate(rows):
        p_plus = first_probs[i, sign_ids["+"]].item()
        p_minus = first_probs[i, sign_ids["-"]].item()
        raw = tokenizer.decode(generated[i], skip_special_tokens=True)
        results.append(
            {
                "game_id": row["game_id"],
                "truth": row["completion"],
                "raw_output": raw,
                "raw_output_tokens": "|".join(tokenizer.decode([t]) for t in generated[i]),
                "parsed": ClassificationTester.post_process(raw),
                "p_plus": p_plus,
                "p_minus": p_minus,
                # Mass on the two signs: near 1 means the model always opens
                # with a sign, as trained; low means it opens with something else.
                "p_sign_mass": p_plus + p_minus,
                # Home-win probability, renormalised over the two signs.
                "p_home": p_plus / (p_plus + p_minus) if p_plus + p_minus > 0 else float("nan"),
                "top_first_tokens": ", ".join(
                    f"{tokenizer.decode([t])!r}:{p:.3f}"
                    for t, p in zip(top_ids[i].tolist(), top_p[i].tolist(), strict=True)
                ),
            }
        )
    return results


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    """Aggregate the per-game frame into the numbers that locate the fault.

    Needs no GPU, so it can be re-run on a saved CSV.
    """
    from sklearn.metrics import roc_auc_score

    truth = frame["truth"].map(lambda c: ground_truth({"completion": c}))
    home_won = truth > 0
    first_char = frame["raw_output"].astype(str).str.lstrip().str[:1]

    summary: dict[str, Any] = {
        "n": len(frame),
        "truth_home_win_rate": home_won.mean(),
        "raw_starts_with_plus": (first_char == "+").mean(),
        "raw_starts_with_minus": (first_char == "-").mean(),
        "raw_starts_with_other": (~first_char.isin(SIGNS)).mean(),
        "parsed_negative_rate": (frame["parsed"] < 0).mean(),
        "mean_p_sign_mass": frame["p_sign_mass"].mean(),
        "p_home_quantiles": frame["p_home"].quantile([0.0, 0.1, 0.5, 0.9, 1.0]).round(3).to_dict(),
        "accuracy_greedy": ((frame["parsed"] > 0) == home_won).mean(),
    }
    if home_won.nunique() == 2:
        # Ranking quality, independent of where the 0.5 threshold sits.
        summary["auc_p_home"] = roc_auc_score(home_won, frame["p_home"])
        # Accuracy if the threshold were moved to the median p_home, i.e. if
        # the model's bias towards "+" were removed and only its ranking kept.
        summary["accuracy_at_median_threshold"] = (
            (frame["p_home"] > frame["p_home"].median()) == home_won
        ).mean()
    return summary


def inspect_adapter(
    config,
    run_name: str,
    split: str,
    size: int,
    batch_size: int,
    revision=None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Run the adapter for *run_name* over *size* rows of *split*.

    Returns the per-game frame, its :func:`summarize` summary and a
    :func:`tokenization_check` of the first row.
    """
    from tqdm.auto import tqdm

    from src.ml.llm.dataset import load_hf_split
    from src.ml.llm.evaluate import _load_adapter_model
    from src.ml.llm.finetune import build_hub_model_id

    data = load_hf_split(config, split)
    size = min(size, len(data))
    tokenizer, model = _load_adapter_model(
        config, build_hub_model_id(config, run_name), revision=revision
    )

    first = data[0]
    tok_check = tokenization_check(tokenizer, first[config.data.prompt_field], first["completion"])

    tokenizer.padding_side = "left"
    records = []
    for start in tqdm(range(0, size, batch_size)):
        rows = [data[i] for i in range(start, min(start + batch_size, size))]
        records.extend(
            predict_with_first_token_probs(
                model,
                tokenizer,
                rows,
                prompt_field=config.data.prompt_field,
                max_new_tokens=config.evaluation.max_new_tokens,
            )
        )

    frame = pd.DataFrame(records)
    return frame, summarize(frame), tok_check
