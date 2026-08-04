"""The LLM prompt must carry every feature and none of the outcome.

``serialize_row`` iterates whatever columns the feature frame happens to have,
which is what keeps it in step with configs/features.yaml — and also exactly
where a leaked target column would slip through unnoticed. These tests pin both
halves of that bargain.
"""

import json
import re

import pandas as pd
import pytest

from src.ml.llm.serialization import (
    LEAKING_COLUMNS,
    PROMPT_SUFFIX,
    serialize_frame,
    serialize_row,
)

FORMATS = ["json", "markdown"]


def emitted_feature_names(text: str, fmt: str) -> set[str]:
    """Feature names actually present in a prompt.

    Compared as names rather than substrings: ``pts_diff_avg_L5_delta`` is a
    legitimate feature that contains the leaking column ``pts_diff`` as a
    substring, so a naive ``in`` check would fire on healthy output.
    """
    if fmt == "markdown":
        rows = re.findall(r"^\|\s*(\S+)\s*\|\s*.*\|$", text, flags=re.MULTILINE)
        return {name for name in rows if name not in ("feature", "---")}

    start = text.index("{")
    payload, _ = json.JSONDecoder().raw_decode(text[start:])
    return set(payload["features"])


@pytest.fixture
def features() -> pd.Series:
    return pd.Series(
        {
            "record_L82_delta": 0.125,
            "pts_diff_avg_L5_delta": -3.5,
            "rested_days_delta": 1.0,
            "east_wins_pct_L1": 0.48,
            "days_at_home": 4,
        }
    )


@pytest.fixture
def meta() -> pd.Series:
    return pd.Series(
        {
            "gameId": 22200001,
            "hometeamName": "Bulls",
            "awayteamName": "76ers",
            "hometeamConference": "East",
            "awayteamConference": "East",
            "season": "2022/23",
            "gameDateOnlyStr": "2022-10-29",
            "is_neutral_court_game": False,
            # Outcome columns that must never reach the prompt:
            "winner": 1,
            "homeScore": 112,
            "awayScore": 99,
            "pts_diff": 13,
        }
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_every_feature_appears(features, meta, fmt):
    """No silent drops — a feature missing from the prompt is invisible damage."""
    assert emitted_feature_names(serialize_row(features, meta, fmt=fmt)  , fmt) == set(
        features.index
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_no_outcome_leakage(features, meta, fmt):
    """The prompt must not contain the answer, by column name or by value."""
    text = serialize_row(features, meta, fmt=fmt)

    assert emitted_feature_names(text, fmt).isdisjoint(LEAKING_COLUMNS)

    # The scores and margin themselves must not appear either.
    body = text.replace(PROMPT_SUFFIX, "")
    for value in ("112", "99"):
        assert value not in body, f"leaked score {value} in {fmt} prompt"


@pytest.mark.parametrize("fmt", FORMATS)
def test_leaking_feature_column_is_stripped(features, meta, fmt):
    """Even if an outcome column reaches the feature frame, it is filtered out."""
    contaminated = pd.concat([features, pd.Series({"pts_diff": 13.0})])
    names = emitted_feature_names(serialize_row(contaminated, meta, fmt=fmt), fmt)

    assert "pts_diff" not in names
    # ...while the similarly-named legitimate feature survives.
    assert "pts_diff_avg_L5_delta" in names


@pytest.mark.parametrize("fmt", FORMATS)
def test_prompt_ends_ready_for_completion(features, meta, fmt):
    assert serialize_row(features, meta, fmt=fmt).endswith(PROMPT_SUFFIX)


@pytest.mark.parametrize("fmt", FORMATS)
def test_context_included(features, meta, fmt):
    text = serialize_row(features, meta, fmt=fmt)
    assert "Bulls" in text and "76ers" in text and "2022/23" in text


def test_serialize_row_without_metadata(features):
    """Inference rows may arrive without metadata; that must still serialize."""
    text = serialize_row(features, None, fmt="markdown")
    assert "record_L82_delta" in text
    assert text.endswith(PROMPT_SUFFIX)


def test_unknown_format_rejected(features, meta):
    with pytest.raises(ValueError, match="Unknown serialization_format"):
        serialize_row(features, meta, fmt="yaml")


def test_serialize_frame_aligns_metadata_by_index(features, meta):
    """Rows must not be paired with the wrong game's metadata."""
    X = pd.DataFrame([features, features * 2], index=[7, 3])
    meta_df = pd.DataFrame(
        [meta, meta.copy()], index=[3, 7]  # deliberately reversed
    )
    meta_df.loc[7, "hometeamName"] = "Lakers"
    meta_df.loc[3, "hometeamName"] = "Celtics"

    texts = serialize_frame(X, meta_df, fmt="markdown")

    assert "Lakers" in texts[0], "row index 7 should carry index 7's metadata"
    assert "Celtics" in texts[1], "row index 3 should carry index 3's metadata"


def test_prose_format_reports_missing_columns_clearly(features, meta):
    """The prose template is pinned to old columns; the failure must explain itself."""
    with pytest.raises(KeyError, match="prose"):
        serialize_row(features, meta, fmt="prose")
