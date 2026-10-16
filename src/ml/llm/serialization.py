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

from src.config.constants import RESPONSE_VARIABLES

SerializationFormat = Literal["labeled", "json", "markdown", "per_team"]

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
        "total_wins_HT",
        "total_losses_HT",
        "total_wins_VT",
        "total_losses_VT",
    }
    | set(RESPONSE_VARIABLES)
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
# has to carry its own meaning. `margin_scaled_avg_L8_delta` means nothing to a
# model pretrained on prose; "normalized point differential, last 8 games" leans
# on what it already knows. These labels decode the column-naming convention
# rather than enumerating columns, so a feature added in configs/features.yaml
# still reaches the prompt with no change here.

_STEM_LABELS = {
    "margin_scaled_avg": "normalized point differential",
    "margin_avg": "point differential",
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
    ("Recent form", ("margin_scaled_avg", "margin_avg", "record", "gds", "streak")),
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


# --- 'per_team' format ------------------------------------------------------
#
# Renders the layout in prompt_template_per_team.md. Unlike the formats above,
# it does not iterate whatever columns exist: it reads a fixed, explicit set of
# columns from the *raw* feature-table row (games_enriched), which still holds
# both teams' values and the games-played counters the per-game rules need.
# Which of those features appear, and as team rows or a home − away delta, is
# chosen by the prompt config (configs/llm_prompt/*.yaml).

PER_TEAM_HEADER = (
    "Predict the final NBA score margin (home − away) from pre-game stats.\n"
    "Adj Win% = win percentage adjusted for strength of schedule, in percent. "
    "Ln = last n games.\n"
    "Scaled Margin = team's avg point differential in hundredths of the season's "
    "margin standard deviation, so 100 = one std (positive = outscored opponents)."
)
PER_TEAM_SUFFIX = "Home margin ="

# Single-value features: key -> (label, home column, away column, value kind).
# Display order is the order here.
PER_TEAM_SCALARS = {
    "streak": ("Streak", "streak_HT", "streak_VT", "signed"),
    "venue_streak": ("Venue streak", "streak_HT_at_home", "streak_VT_on_road", "signed"),
    # Home team's days at home, away team's days on the road.
    "days_at_venue": ("Days at venue", "days_at_home", "days_on_road", "int"),
    "rest_days": ("Rest days", "rested_days_HT", "rested_days_VT", "int"),
    "back_to_back": ("Back-to-back", "back_to_back_HT", "back_to_back_VT", "int"),
    "playoffs": ("Playoffs", None, None, "playoffs"),
    "last_season_record": (
        "Adj Last-season win%",
        "adjusted_last_season_record_HT",
        "adjusted_last_season_record_VT",
        "pct",
    ),
    "last_season_venue_record": (
        "Adj Last-season venue win%",
        "adjusted_last_season_record_HT_at_home",
        "adjusted_last_season_record_VT_on_road",
        "pct",
    ),
}

# Lag tables: key -> (label, home column, away column, windows, column header,
# value kind, games counter used to bound the windows). `{n}` is the window.
# A None counter means the windows are days, not games, so they are never cut.
_OVERALL = ("games_played_HT", "games_played_VT")
_VENUE = ("games_played_HT_at_home", "games_played_VT_on_road")
_GAME_LAGS = (3, 5, 8, 13, 21, 34, 55, 82)
_VENUE_LAGS = (5, 10, 20, 41)

PER_TEAM_TABLES = {
    "win_pct": (
        "Adj Win%", "sos_adj_record_L{n}_HT", "sos_adj_record_L{n}_VT",
        _GAME_LAGS, "L{n}", "pct", _OVERALL,
    ),
    "pt_diff": (
        "Scaled Margin", "margin_scaled_avg_L{n}_HT", "margin_scaled_avg_L{n}_VT",
        _GAME_LAGS, "L{n}", "pct", _OVERALL,
    ),
    "venue_win_pct": (
        "Adj Venue win%", "sos_adj_record_L{n}_HT_at_home", "sos_adj_record_L{n}_VT_on_road",
        _VENUE_LAGS, "L{n}", "pct", _VENUE,
    ),
    "venue_pt_diff": (
        "Scaled Margin Venue", "margin_scaled_avg_L{n}_HT_at_home",
        "margin_scaled_avg_L{n}_VT_on_road", _VENUE_LAGS, "L{n}", "pct", _VENUE,
    ),
    "travel": (
        "Avg km/day", "distance_L{n}_HT", "distance_L{n}_VT",
        (1, 3, 7, 14), "{n}d", "int", None,
    ),
}

# The conference line is its own sentence, not a table row.
CONFERENCE_MATCHUP = "conference_matchup"

# A difference of these means nothing, so they only render as team values.
_TEAM_ONLY = {"playoffs", CONFERENCE_MATCHUP}

PER_TEAM_FEATURES = (*PER_TEAM_SCALARS, *PER_TEAM_TABLES, CONFERENCE_MATCHUP)


def load_prompt_config(path) -> dict:
    """Read and check a prompt config (configs/llm_prompt/*.yaml)."""
    from src.ml.config.loader import load_yaml_config

    config = load_yaml_config(path)
    for key, mode in config.get("features", {}).items():
        if key not in PER_TEAM_FEATURES:
            raise ValueError(
                f"Unknown prompt feature '{key}' in {path}. "
                f"Expected one of {list(PER_TEAM_FEATURES)}."
            )
        if mode not in ("team", "delta"):
            raise ValueError(f"Prompt feature '{key}' in {path}: mode must be team or delta, got '{mode}'.")
        if mode == "delta" and key in _TEAM_ONLY:
            raise ValueError(f"Prompt feature '{key}' in {path} can only be shown as 'team'.")
    for name in ("last_season_min_games", "last_season_venue_min_games", "playoffs_min_games"):
        if name not in config.get("thresholds", {}):
            raise ValueError(f"Prompt config {path} is missing thresholds.{name}.")
    return config


def bound_lags(lags: tuple[int, ...], games: int) -> tuple[int, ...]:
    """Every window shorter than ``games``, plus the first that covers them all.

    With 19 (or 21) games played, L21 already holds the whole season; L34 and
    beyond would repeat it. With 22 games, L34 adds the one game L21 misses.
    """
    covering = [n for n in lags if n >= games]
    return tuple(n for n in lags if n < games) + tuple(covering[:1])


def hidden_features(raw: pd.Series, thresholds: dict) -> set[str]:
    """Features the per-game rules drop for this game, whatever the config says."""
    games = min(raw["games_played_HT"], raw["games_played_VT"])
    venue_games = min(raw["games_played_HT_at_home"], raw["games_played_VT_on_road"])

    hidden = set()
    if games >= thresholds["last_season_min_games"]:
        hidden.add("last_season_record")
    if venue_games >= thresholds["last_season_venue_min_games"]:
        hidden.add("last_season_venue_record")
    if not games > thresholds["playoffs_min_games"]:
        hidden.add("playoffs")
    if raw["hometeamConference"] == raw["awayteamConference"]:
        hidden.add(CONFERENCE_MATCHUP)
    return hidden


def _playoff_status(raw: pd.Series, team: str) -> str:
    """First flag that holds, in order: seed locked, clinched, eliminated."""
    if raw[f"clinched_final_seed_{team}"]:
        return "seed locked"
    if raw[f"clinched_playoff_berth_{team}"]:
        return "clinched"
    if raw[f"eliminated_from_playoffs_{team}"]:
        return "eliminated"
    return "in race"


def _format_cell(value, kind: str, signed: bool = False) -> str:
    """``pct`` values are fractions shown as whole percent; every number is an integer."""
    if pd.isna(value):
        return "-"
    number = round(float(value) * 100) if kind == "pct" else round(float(value))
    if (signed or kind == "signed") and number > 0:
        return f"+{number}"
    return str(number)


def _grid(rows: list[list[str]]) -> list[str]:
    """First column left-aligned, the rest right-aligned, two spaces apart."""
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return [
        (row[0].ljust(widths[0]) + "".join(f"  {c.rjust(w)}" for c, w in zip(row[1:], widths[1:], strict=True))).rstrip()
        for row in rows
    ]


def _scalar_blocks(raw: pd.Series, shown: dict[str, str]) -> list[str]:
    team_rows, delta_rows = [], []
    for key, (label, home_col, away_col, kind) in PER_TEAM_SCALARS.items():
        if key not in shown:
            continue
        if kind == "playoffs":
            team_rows.append([label, _playoff_status(raw, "HT"), _playoff_status(raw, "VT")])
        elif shown[key] == "team":
            team_rows.append([label, _format_cell(raw[home_col], kind), _format_cell(raw[away_col], kind)])
        else:
            delta_rows.append([label, _format_cell(raw[home_col] - raw[away_col], kind, signed=True)])

    blocks = []
    if team_rows:
        blocks.append("\n".join(_grid([["Team", "Home", "Away"], *team_rows])))
    if delta_rows:
        blocks.append("\n".join(_grid([["Team", "Home−Away"], *delta_rows])))
    return blocks


def _lag_table(raw: pd.Series, key: str, mode: str) -> str:
    label, home_col, away_col, lags, head, kind, counter = PER_TEAM_TABLES[key]
    if counter is not None:
        # The more-played team decides, so neither team loses a window it needs.
        lags = bound_lags(lags, int(max(raw[counter[0]], raw[counter[1]])))

    home = [raw[home_col.format(n=n)] for n in lags]
    away = [raw[away_col.format(n=n)] for n in lags]

    rows = [[label, *(head.format(n=n) for n in lags)]]
    if mode == "team":
        rows.append(["Home", *(_format_cell(v, kind) for v in home)])
        rows.append(["Away", *(_format_cell(v, kind) for v in away)])
    else:
        rows.append(
            ["Home−Away", *(_format_cell(h - a, kind, signed=True) for h, a in zip(home, away, strict=True))]
        )
    return "\n".join(_grid(rows))


def _conference_line(raw: pd.Series) -> Optional[str]:
    conf = str(raw["hometeamConference"]).lower()
    if conf not in ("east", "west"):
        return None
    overall = _format_cell(raw[f"{conf}_record_adjusted"], "pct")
    floor = _format_cell(raw[f"{conf}_record_at_{conf}"], "pct")
    return (
        "Conference matchup this season (home conf vs away conf): "
        f"{overall}% overall, {floor}% on home-conf floor"
    )


def _as_per_team(raw: pd.Series, prompt_config: dict) -> str:
    """The whole prompt, header to ``PER_TEAM_SUFFIX``."""
    hidden = hidden_features(raw, prompt_config["thresholds"])
    shown = {k: m for k, m in prompt_config.get("features", {}).items() if k not in hidden}

    blocks = [PER_TEAM_HEADER, *_scalar_blocks(raw, shown)]
    blocks.extend(_lag_table(raw, key, shown[key]) for key in PER_TEAM_TABLES if key in shown)
    if CONFERENCE_MATCHUP in shown:
        line = _conference_line(raw)
        if line is not None:
            blocks.append(line)
    blocks.append(PER_TEAM_SUFFIX)
    return "\n\n".join(blocks)


_SERIALIZERS = {
    "labeled": _as_labeled,
    "json": _as_json,
    "markdown": _as_markdown,
}


def serialize_row(
    features: pd.Series,
    meta: Optional[pd.Series] = None,
    fmt: SerializationFormat = "markdown",
    raw: Optional[pd.Series] = None,
    prompt_config: Optional[dict] = None,
) -> str:
    """Render one feature row as the prompt text shown to the model.

    Parameters
    ----------
    features : pd.Series
        One row of the post-feature-engineering ``X`` frame.
    meta : pd.Series, optional
        Matching row of the metadata frame (team names, season, date). Only the
        non-leaking columns in ``CONTEXT_META_COLUMNS`` are used.
    fmt : {'labeled', 'json', 'markdown', 'per_team'}
        Encoding. The first three follow whatever columns exist. ``labeled``
        additionally renders human-readable feature names, which a base
        (non-instruct) model has more to work with. ``per_team`` ignores
        ``features`` and renders a fixed set of columns from ``raw`` as laid
        out in prompt_template_per_team.md.
    raw : pd.Series, optional
        Matching row of the untouched feature table (``Splits.games_enriched``).
        Required for ``per_team``.
    prompt_config : dict, optional
        Output of :func:`load_prompt_config`. Required for ``per_team``.

    Returns
    -------
    str
        Prompt text ending in ``PROMPT_SUFFIX`` (``PER_TEAM_SUFFIX`` for
        ``per_team``), ready for a completion.
    """
    if fmt == "per_team":
        if raw is None or prompt_config is None:
            raise ValueError("serialization_format 'per_team' needs both `raw` and `prompt_config`.")
        return _as_per_team(raw, prompt_config)

    if fmt not in _SERIALIZERS:
        raise ValueError(
            f"Unknown serialization_format '{fmt}'. "
            f"Expected one of {sorted([*_SERIALIZERS, 'per_team'])}."
        )

    body = _SERIALIZERS[fmt](features, meta)
    return f"{SYSTEM_PROMPT}\n{body}\n\n{QUESTION}{PROMPT_SUFFIX}"


def serialize_frame(
    features: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
    fmt: SerializationFormat = "markdown",
    raw: Optional[pd.DataFrame] = None,
    prompt_config: Optional[dict] = None,
) -> list[str]:
    """Serialize every row of ``features``, aligning ``metadata`` and ``raw`` by index."""
    if metadata is not None:
        metadata = metadata.reindex(features.index)
    if raw is not None:
        raw = raw.reindex(features.index)

    return [
        serialize_row(
            features.loc[idx],
            metadata.loc[idx] if metadata is not None else None,
            fmt=fmt,
            raw=raw.loc[idx] if raw is not None else None,
            prompt_config=prompt_config,
        )
        for idx in features.index
    ]
