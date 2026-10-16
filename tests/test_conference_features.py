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
    "games_played_east_vs_west",
    "last_season_east_record_adjusted",
    "last_season_west_record_adjusted",
    "last_season_east_record_at_east",
    "last_season_west_record_at_west",
]


def _frame(**overrides) -> pd.DataFrame:
    """Four games: two cross-conference (both directions) and two intra.

    ``games_played_east_vs_west`` defaults well above the default
    ``saturation_games`` (20) and ``last_season_*`` default to NaN, so the
    blend ramp lands at full weight on the current-season columns and every
    pre-existing assertion below keeps holding unchanged.
    """
    base = {
        "hometeamConference": ["East", "West", "East", "West"],
        "awayteamConference": ["West", "East", "East", "West"],
        "east_record_adjusted": [0.55, 0.55, 0.55, 0.55],
        "west_record_adjusted": [0.45, 0.45, 0.45, 0.45],
        "east_record_at_east": [0.60, 0.60, 0.60, 0.60],
        "west_record_at_west": [0.52, 0.52, 0.52, 0.52],
        "games_played_east_vs_west": [999, 999, 999, 999],
        "last_season_east_record_adjusted": [float("nan")] * 4,
        "last_season_west_record_adjusted": [float("nan")] * 4,
        "last_season_east_record_at_east": [float("nan")] * 4,
        "last_season_west_record_at_west": [float("nan")] * 4,
        "record_L10_delta": [0.1, -0.2, 0.3, 0.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


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


def test_blend_uses_last_season_fully_at_zero_games_played():
    """0 interconference games played this season -> ramp weight is 0."""
    frame = _frame(
        games_played_east_vs_west=[0, 0, 0, 0],
        last_season_east_record_adjusted=[0.70, 0.70, 0.70, 0.70],
        last_season_west_record_adjusted=[0.30, 0.30, 0.30, 0.30],
        last_season_east_record_at_east=[0.80, 0.80, 0.80, 0.80],
        last_season_west_record_at_west=[0.20, 0.20, 0.20, 0.20],
    )
    out = create_conference_features(frame, saturation_games=20)

    # Row 0: East hosts West. diff = +1 * (0.70 - 0.30) = 0.40.
    assert out.loc[0, "conference_diff_home_advantage_pct"] == pytest.approx(0.40)
    # Row 0: home court uses last season's east_record_at_east, centered.
    assert out.loc[0, "conference_home_court_advantage_pct"] == pytest.approx(0.30)
    # Row 1: West hosts East. home court uses last season's west_record_at_west.
    assert out.loc[1, "conference_home_court_advantage_pct"] == pytest.approx(-0.30)


def test_blend_ramps_linearly_between_last_and_current_season():
    """Midway through the ramp, the blend is the weighted average of both."""
    frame = _frame(
        games_played_east_vs_west=[10, 10, 10, 10],  # half of saturation_games=20
        last_season_east_record_adjusted=[0.70, 0.70, 0.70, 0.70],
        last_season_west_record_adjusted=[0.30, 0.30, 0.30, 0.30],
    )
    out = create_conference_features(frame, saturation_games=20)

    # weight = 0.5 -> east_eff = 0.5*0.55 + 0.5*0.70 = 0.625, west_eff = 0.5*0.45 + 0.5*0.30 = 0.375
    # Row 0: East hosts West. diff = +1 * (0.625 - 0.375) = 0.25.
    assert out.loc[0, "conference_diff_home_advantage_pct"] == pytest.approx(0.25)


def test_blend_falls_back_to_current_season_when_last_season_missing():
    """First season in the dataset: no prior season to blend with.

    Even with a thin current-season sample, NaN last-season inputs must not
    propagate into the feature — the row falls back to the current-season
    value outright.
    """
    frame = _frame(games_played_east_vs_west=[1, 1, 1, 1])
    out = create_conference_features(frame, saturation_games=20)

    assert not out["conference_diff_home_advantage_pct"].isna().any()
    assert not out["conference_home_court_advantage_pct"].isna().any()
    # Same as the full-weight case: last_season_* is NaN, so the raw
    # current-season values pass through unblended.
    assert out.loc[0, "conference_diff_home_advantage_pct"] == pytest.approx(0.10)
