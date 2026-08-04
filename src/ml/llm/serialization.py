"""Turn one tabular feature row into prompt text for the LLM.

This is the *only* place the table becomes text. Training, validation, testing
and inference all route through :func:`serialize_row`, so the LLM can never
drift onto a second feature path.

The row handed in is the post-feature-engineering ``X`` produced by
``src/ml/datasets/splits.py`` — the very same columns the sklearn models see.
That is deliberate: it makes "did the LLM beat the ML models?" a fair question,
and it means a feature added in ``configs/features.yaml`` reaches the prompt
with no code change here.
"""

from __future__ import annotations

import json
from typing import Literal, Optional

import pandas as pd

SerializationFormat = Literal["json", "markdown", "prose"]

SYSTEM_PROMPT = (
    "You are responsible for predicting the outcome of a basketball game.\n"
    "Below are the pre-game statistics for the home and visiting teams. "
    "Positive values favour the home team.\n"
)

QUESTION = (
    "What is the point differential between the home team and the visiting team?"
    " Use positive values for the home team winning the match and negative values"
    " for the visiting team winning it.\n"
)

PROMPT_SUFFIX = "The home team finished the game with a point differential of"

# Never allowed into a prompt: these encode the result we are asking the model
# to predict. Guarded here rather than upstream because a serializer that
# iterates "whatever columns exist" is exactly where leakage would slip in.
LEAKING_COLUMNS = frozenset(
    {
        "winner",
        "winnerteamConference",
        "homeScore",
        "awayScore",
        "pts_diff",
        "win_bool",
        "total_wins_HT",
        "total_losses_HT",
        "total_wins_VT",
        "total_losses_VT",
    }
)

# Contextual metadata that is safe and genuinely useful to state in the prompt.
CONTEXT_META_COLUMNS = (
    "hometeamName",
    "awayteamName",
    "hometeamConference",
    "awayteamConference",
    "season",
    "gameDateOnlyStr",
    "is_neutral_court_game",
)


def _format_value(value) -> str:
    if pd.isna(value):
        return "unknown"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _safe_features(features: pd.Series) -> pd.Series:
    """Drop any outcome-bearing column that reached the feature frame."""
    return features.drop(labels=[c for c in features.index if c in LEAKING_COLUMNS])


def _context(meta: Optional[pd.Series]) -> dict:
    if meta is None:
        return {}
    return {
        col: _format_value(meta[col])
        for col in CONTEXT_META_COLUMNS
        if col in meta.index and col not in LEAKING_COLUMNS
    }


def _as_json(features: pd.Series, meta: Optional[pd.Series]) -> str:
    payload = {
        "game": _context(meta),
        "features": {name: _format_value(v) for name, v in _safe_features(features).items()},
    }
    return json.dumps(payload, indent=2, sort_keys=False)


def _as_markdown(features: pd.Series, meta: Optional[pd.Series]) -> str:
    lines = []

    context = _context(meta)
    if context:
        lines.append("## Game")
        lines.extend(f"- {key}: {value}" for key, value in context.items())
        lines.append("")

    lines.append("## Pre-game features")
    lines.append("| feature | value |")
    lines.append("| --- | --- |")
    lines.extend(
        f"| {name} | {_format_value(value)} |" for name, value in _safe_features(features).items()
    )
    return "\n".join(lines)


def _as_prose(features: pd.Series, meta: Optional[pd.Series]) -> str:
    """The original hand-written prompt, via :class:`src.ml.llm.prompts.Game`.

    Retained so runs already on the Hub stay comparable. It requires a fixed
    set of raw ``*_HT``/``*_VT`` columns, so it only works on feature frames
    that still carry them — see :func:`src.ml.llm.prompts.build_prose_prompt`.
    """
    from src.ml.llm.prompts import build_prose_prompt

    return build_prose_prompt(features, meta)


_SERIALIZERS = {
    "json": _as_json,
    "markdown": _as_markdown,
    "prose": _as_prose,
}


def serialize_row(
    features: pd.Series,
    meta: Optional[pd.Series] = None,
    fmt: SerializationFormat = "markdown",
) -> str:
    """Render one feature row as the prompt text shown to the model.

    Parameters
    ----------
    features : pd.Series
        One row of the post-feature-engineering ``X`` frame.
    meta : pd.Series, optional
        Matching row of the metadata frame (team names, season, date). Only the
        non-leaking columns in ``CONTEXT_META_COLUMNS`` are used.
    fmt : {'json', 'markdown', 'prose'}
        Encoding. ``markdown`` and ``json`` follow whatever columns exist;
        ``prose`` uses the fixed original template.

    Returns
    -------
    str
        Prompt text ending in ``PROMPT_SUFFIX``, ready for a completion.
    """
    if fmt not in _SERIALIZERS:
        raise ValueError(
            f"Unknown serialization_format '{fmt}'. Expected one of {sorted(_SERIALIZERS)}."
        )

    body = _SERIALIZERS[fmt](features, meta)
    return f"{SYSTEM_PROMPT}\n{body}\n\n{QUESTION}{PROMPT_SUFFIX}"


def serialize_frame(
    features: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
    fmt: SerializationFormat = "markdown",
) -> list[str]:
    """Serialize every row of ``features``, aligning ``metadata`` by index."""
    if metadata is not None:
        metadata = metadata.reindex(features.index)

    return [
        serialize_row(
            features.loc[idx],
            metadata.loc[idx] if metadata is not None else None,
            fmt=fmt,
        )
        for idx in features.index
    ]
