"""create_conference_features must emit exactly what resolve_feature_columns declares.

The two functions mirror each other: one builds the conference columns, the
other names them so inclusion-mode selection can keep them. Nothing asserted
that mirror from the building side — renaming a column inside the builder would
have left the feature silently missing at train time and only surfaced as a
warning buried in the log.

The rest of this file pins the off-population rule. Both features estimate a
conference-vs-conference advantage, which exists only for interconference
games; on a same-conference game the effect cancels exactly, so both are 0.0.
Leaving the second one ungated would be a confound rather than noise — its
ingredients are defined on every date, so every East home team would get one
value and every West home team another.
"""

import pandas as pd
import pytest

from src.ml.config.schema import FeatureGroupConfig, FeaturesMapConfig
from src.ml.features.engineering import (
    create_conference_features,
    resolve_feature_columns,
)

# Columns the east/west ETL tables contribute, all of which are intermediates:
# the builder is expected to consume every one of them.
RAW_CONFERENCE_COLUMNS = [
    "east_record_adjusted",
    "west_record_adjusted",
    "east_record_at_east",
    "west_record_at_west",
]


def _frame() -> pd.DataFrame:
    """Four games: two cross-conference (both directions) and two intra."""
    return pd.DataFrame(
        {
            "hometeamConference": ["East", "West", "East", "West"],
            "awayteamConference": ["West", "East", "East", "West"],
            "east_record_adjusted": [0.55, 0.55, 0.55, 0.55],
            "west_record_adjusted": [0.45, 0.45, 0.45, 0.45],
            "east_record_at_east": [0.60, 0.60, 0.60, 0.60],
            "west_record_at_west": [0.52, 0.52, 0.52, 0.52],
            "record_L10_delta": [0.1, -0.2, 0.3, 0.0],
        }
    )


def _declared_conference_columns() -> list[str]:
    """The conference columns resolve_feature_columns asks for, and only those."""
    only_record = FeaturesMapConfig(
        **{
            name: FeatureGroupConfig(enabled=False, lags=[])
            for name in FeaturesMapConfig.model_fields
        }
        | {"record": FeatureGroupConfig(lags=[10], delta=True, enabled=True)}
    )
    resolved = resolve_feature_columns(only_record)
    return [column for column in resolved if column != "record_L10_delta"]


def test_every_declared_conference_column_is_built():
    out = create_conference_features(_frame())
    missing = [
        column for column in _declared_conference_columns() if column not in out.columns
    ]
    assert not missing, (
        f"resolve_feature_columns declares {missing} but "
        "create_conference_features does not build them"
    )


def test_raw_conference_intermediates_are_consumed():
    """None of the east/west inputs may survive as a feature.

    They are conference-season-level, so on a same-conference game they describe
    a contest that is not happening.
    """
    out = create_conference_features(_frame())
    leaked = [column for column in RAW_CONFERENCE_COLUMNS if column in out.columns]
    assert not leaked, f"intermediates left behind: {leaked}"


def test_conference_diff_is_zero_for_intra_conference_games():
    """The claim that makes one model over all games plausible in the first place."""
    frame = _frame()
    out = create_conference_features(frame)
    intra = frame["hometeamConference"] == frame["awayteamConference"]

    assert (out.loc[intra, "conference_diff_home_advantage_pct"] == 0.0).all()
    assert (out.loc[~intra, "conference_diff_home_advantage_pct"] != 0.0).all()
    # East at home against a West visitor reads positive when East is stronger.
    assert out.loc[0, "conference_diff_home_advantage_pct"] == pytest.approx(0.10)
    assert out.loc[1, "conference_diff_home_advantage_pct"] == pytest.approx(-0.10)


def test_home_court_advantage_reads_the_home_conference_side_centered():
    """Cross-conference: the home team's own conference's home rate, minus 0.5."""
    out = create_conference_features(_frame())
    assert out.loc[0, "conference_home_court_advantage_pct"] == pytest.approx(0.10)
    assert out.loc[1, "conference_home_court_advantage_pct"] == pytest.approx(0.02)


def test_home_court_advantage_is_zero_for_intra_conference_games():
    """Not 0.60 or 0.52 — the rate describes a matchup that is not happening.

    Centering is what lets "no effect" and "no data" be the same number. Without
    the gate, rows 2 and 3 would inherit their conference's rate and hand the
    model a constant-per-conference confound.
    """
    frame = _frame()
    out = create_conference_features(frame)
    intra = frame["hometeamConference"] == frame["awayteamConference"]

    assert (out.loc[intra, "conference_home_court_advantage_pct"] == 0.0).all()
