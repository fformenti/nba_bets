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
import re
from typing import Literal, Optional

import pandas as pd

SerializationFormat = Literal["labeled", "json", "markdown"]

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


# --- 'labeled' format -------------------------------------------------------
#
# The base Llama checkpoint has no instruction tuning to lean on, so the prompt
# has to carry its own meaning. `norm_pts_diff_avg_L8_delta` means nothing to a
# model pretrained on prose; "normalized point differential, last 8 games" leans
# on what it already knows. These labels decode the column-naming convention
# rather than enumerating columns, so a feature added in configs/features.yaml
# still reaches the prompt with no change here.

_STEM_LABELS = {
    "norm_pts_diff_avg": "normalized point differential",
    "pts_diff_avg": "point differential",
    "record": "win percentage",
    "sos_adj_record": "strength-of-schedule adjusted win percentage",
    "sos": "strength of schedule",
    "gds": "game difficulty score",
    "distance": "travel distance",
    "rested_days": "days of rest",
    "back_to_back": "playing on no rest",
    "streak": "current win/loss streak",
    "last_season_record": "last season win percentage",
    "adjusted_last_season_record": "adjusted last-season win percentage",
    "days_at_home": "consecutive days at home",
    "days_on_road": "consecutive days on the road",
    "conference_diff_home_advantage_pct": "cross-conference home advantage",
    "conference_home_court_advantage_pct": "home conference's edge on its own floor",
    "winning_incentive": (
        "what the team still has to play for "
        "(-1 eliminated, 0 seed locked in and resting, 1 still fighting)"
    ),
}

# A group with `delta: false` ships its raw per-team pair, so the column keeps a
# team suffix that means nothing to a model reading prose.
_TEAM_SUFFIXES = {"_HT": "home team", "_VT": "away team"}

# (section title, stems). First match wins; anything unmatched lands in "Other",
# so a new feature is never silently dropped from the prompt.
_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Recent form", ("norm_pts_diff_avg", "pts_diff_avg", "record", "gds", "streak")),
    ("Strength of schedule", ("sos_adj_record", "sos")),
    (
        "Rest and travel",
        ("rested_days", "back_to_back", "distance", "days_at_home", "days_on_road"),
    ),
    (
        "Season context",
        (
            "last_season_record",
            "adjusted_last_season_record",
            "conference_diff_home_advantage_pct",
            "conference_home_court_advantage_pct",
        ),
    ),
    (
        "Playoff position",
        (
            "winning_incentive",
            "games_behind_leader",
            "games_behind_above",
            "games_ahead_of_below",
        ),
    ),
)

_VENUE_SECTION = "Venue form (home team at home vs away team on the road)"
_OTHER_SECTION = "Other"


def _parse_feature_name(name: str) -> tuple[str, str, bool]:
    """Split a feature column into ``(stem, human label, is_venue)``.

    Understands the ``<stem>[_L<lag>][_at_location][_delta|_HT|_VT]`` convention.
    An unrecognized stem falls back to its own name with underscores spaced out —
    never raises, so an unfamiliar column degrades to a readable line instead of
    breaking the dataset build.
    """
    stem = name[: -len("_delta")] if name.endswith("_delta") else name

    # A raw (delta: false) column names its team. Say which team in words:
    # "home team" reads; "HT" is a convention only this repo knows.
    team = None
    for suffix, who in _TEAM_SUFFIXES.items():
        if stem.endswith(suffix):
            team = who
            stem = stem[: -len(suffix)]
            break

    is_venue = stem.endswith("_at_location")
    if is_venue:
        stem = stem[: -len("_at_location")]

    lag = None
    match = re.search(r"_L(\d+)$", stem)
    if match:
        lag = int(match.group(1))
        stem = stem[: match.start()]

    label = _STEM_LABELS.get(stem, stem.replace("_", " "))
    if lag is not None:
        label += f", last {lag} game{'s' if lag != 1 else ''}"
    if team is not None:
        label = f"{team} {label}"
    return stem, label, is_venue


def _section_for(stem: str, is_venue: bool) -> str:
    if is_venue:
        return _VENUE_SECTION
    for title, stems in _SECTIONS:
        if stem in stems:
            return title
    return _OTHER_SECTION


def _format_labeled_value(value) -> str:
    """Two decimals, not three: Llama-3 splits digits, so trailing precision on
    a value like -0.005 costs tokens without carrying signal."""
    if pd.isna(value):
        return "unknown"
    if isinstance(value, float):
        text = f"{value:.2f}".rstrip("0").rstrip(".")
        # -0.004 rounds to "-0.00" and strips down to "-0", which reads as a
        # negative quantity when it is just zero.
        return "0" if text in ("", "-0", "-") else text
    return str(value)


def _as_labeled(features: pd.Series, meta: Optional[pd.Series]) -> str:
    grouped: dict[str, list[str]] = {}
    for name, value in _safe_features(features).items():
        stem, label, is_venue = _parse_feature_name(str(name))
        section = _section_for(stem, is_venue)
        grouped.setdefault(section, []).append(
            f"  {label}: {_format_labeled_value(value)}"
        )

    lines = []

    # No per-format sentence about how a delta is computed. This used to say
    # "Each statistic is the home team's value minus the away team's", which is
    # not true of every group: engineering.py::INVERTED_DELTA_GROUPS swaps the
    # operands for lower-is-better quantities (distance, back_to_back) and
    # days_at_home_delta is a sum. One formula cannot describe all three, and a
    # formula per label would cost more tokens than it buys. SYSTEM_PROMPT
    # carries the statement that is true of all of them and is the one the model
    # actually needs — positive values favour the home team.

    context = _context(meta)
    if context:
        # Team names are kept for readability, but note that against an all-delta
        # feature set they invite memorizing team identity across 45 seasons
        # rather than reading the stats. Worth ablating once there are numbers.
        home = context.get("hometeamName", "home team")
        away = context.get("awayteamName", "away team")
        descriptors = [
            d
            for d in (context.get("gameDateOnlyStr"), context.get("season"))
            if d is not None
        ]
        suffix = f" — {', '.join(descriptors)}" if descriptors else ""
        lines.append(f"Game: {home} (home) vs {away} (away){suffix}")
        if context.get("is_neutral_court_game") == "True":
            lines.append("Played on a neutral court.")
        lines.append("")

    ordered = [title for title, _ in _SECTIONS]
    ordered.insert(1, _VENUE_SECTION)
    ordered.append(_OTHER_SECTION)

    for section in ordered:
        if section not in grouped:
            continue
        lines.append(f"{section}:")
        lines.extend(grouped[section])
        lines.append("")

    return "\n".join(lines).rstrip()


_SERIALIZERS = {
    "labeled": _as_labeled,
    "json": _as_json,
    "markdown": _as_markdown,
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
    fmt : {'labeled', 'json', 'markdown'}
        Encoding. All three follow whatever columns exist. ``labeled`` is the
        default: it additionally renders human-readable feature names, which a
        base (non-instruct) model has more to work with.

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
