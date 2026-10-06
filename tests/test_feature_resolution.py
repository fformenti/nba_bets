"""Tests for resolve_feature_columns() and inclusion-mode feature selection."""

import pandas as pd
import pytest

from src.ml.config.schema import (
    FeatureEngineeringConfig,
    FeatureGroupConfig,
    FeaturesMapConfig,
    MomentumPairConfig,
)
from src.ml.features.engineering import (
    CONFERENCE_FEATURES,
    WINNING_INCENTIVE_VALUES,
    create_delta_features,
    create_winning_incentive_features,
    resolve_feature_columns,
)
from src.config.paths import CONFIGS_TRAIN_DIR, TRAIN_DEFAULTS_CONFIG_PATH
from src.ml.config.loader import load_experiment_config

# Absolute, not Path("configs/..."): a cwd-relative constant made these tests
# pass or fail depending on where pytest was invoked from.
TRAIN_ALL = CONFIGS_TRAIN_DIR / "xgboost.yaml"


def _make_disabled() -> FeatureGroupConfig:
    return FeatureGroupConfig(enabled=False, lags=[])


def _all_disabled(**overrides: FeatureGroupConfig) -> FeaturesMapConfig:
    """Every feature group off, except the ones passed in.

    Built from the model fields so groups added to FeaturesMapConfig later
    default to off here instead of silently leaking into resolved columns.
    """
    groups = {name: _make_disabled() for name in FeaturesMapConfig.model_fields}
    groups.update(overrides)
    return FeaturesMapConfig(**groups)


def _record_only(lags: list[int]) -> FeaturesMapConfig:
    """FeaturesMapConfig with only record lags set; all other groups disabled."""
    return _all_disabled(record=FeatureGroupConfig(lags=lags, delta=True, enabled=True))


def _without_conference(cfg: FeaturesMapConfig, **kwargs) -> list[str]:
    """Resolved columns minus the conference pair, which is always appended.

    The conference features are not config-gated — every model gets them, and
    they are 0.0 for same-conference games rather than absent. Group-specific
    tests below assert about their own group, so they subtract them out;
    ``test_conference_features_are_always_resolved`` is what pins them.
    """
    return [
        column
        for column in resolve_feature_columns(cfg, **kwargs)
        if column not in CONFERENCE_FEATURES
    ]


# ── resolve_feature_columns ─────────────────────────────────────────────────


class TestResolveFeatureColumns:
    def test_basic_record_delta(self):
        cfg = _record_only(lags=[10])
        cols = _without_conference(cfg)
        assert cols == ["record_L10_delta"]

    def test_multiple_lags(self):
        cfg = _record_only(lags=[10, 82])
        cols = _without_conference(cfg)
        assert cols == ["record_L10_delta", "record_L82_delta"]

    def test_disabled_group_excluded(self):
        cfg = _all_disabled(record=FeatureGroupConfig(enabled=False, lags=[10]))
        cols = _without_conference(cfg)
        assert cols == []

    def test_sos_adj_record_independent_of_record(self):
        """sos_adj_record with its own lags resolves without needing record intersection."""
        cfg = _all_disabled(
            sos_adj_record=FeatureGroupConfig(enabled=True, lags=[10], delta=True),
        )
        cols = _without_conference(cfg)
        assert "sos_adj_record_L10_delta" in cols
        assert "record_L10_delta" not in cols

    def test_sos_adj_record_and_record_both_enabled(self):
        """Both record and sos_adj_record can be enabled independently."""
        cfg = _all_disabled(
            record=FeatureGroupConfig(enabled=True, lags=[82], delta=True),
            sos_adj_record=FeatureGroupConfig(enabled=True, lags=[13, 55], delta=True),
        )
        cols = _without_conference(cfg)
        assert "record_L82_delta" in cols
        assert "sos_adj_record_L13_delta" in cols
        assert "sos_adj_record_L55_delta" in cols
        assert "record_L13_delta" not in cols

    def test_sos_adj_record_uses_own_lags(self):
        """sos_adj_record resolves only declared lags, not any intersection."""
        cfg = _all_disabled(
            record=FeatureGroupConfig(enabled=True, lags=[1, 3, 5, 82], delta=True),
            sos=FeatureGroupConfig(enabled=False, lags=[1, 3, 5, 82]),
            sos_adj_record=FeatureGroupConfig(enabled=True, lags=[13, 55], delta=True),
        )
        cols = _without_conference(cfg)
        # sos_adj_record produces only its declared lags
        assert "sos_adj_record_L13_delta" in cols
        assert "sos_adj_record_L55_delta" in cols
        assert "sos_adj_record_L1_delta" not in cols
        assert "sos_adj_record_L82_delta" not in cols

    def test_sos_adj_record_location_variants(self):
        """sos_adj_record produces at_location_delta columns when location_lags set."""
        cfg = _all_disabled(
            sos_adj_record=FeatureGroupConfig(
                enabled=True, lags=[13], location_lags=[10, 41], delta=True
            ),
        )
        cols = _without_conference(cfg)
        assert "sos_adj_record_L13_delta" in cols
        assert "sos_adj_record_L10_at_location_delta" in cols
        assert "sos_adj_record_L41_at_location_delta" in cols

    def test_conference_features_are_always_resolved(self):
        """Both features, unconditionally — there is no filter to gate them on."""
        cols = resolve_feature_columns(_record_only(lags=[10]))
        assert "conference_diff_home_advantage_pct" in cols
        assert "conference_home_court_advantage_pct" in cols

    def test_games_played_at_home_conference_is_not_a_feature(self):
        """It is a divisor inside the east/west ETL, not a model input.

        It reached resolve_feature_columns only under the deleted 'different'
        filter, so no shipped config ever fed it to a model.
        """
        cols = resolve_feature_columns(_record_only(lags=[10]))
        assert "games_played_at_home_conference" not in cols

    def test_location_variants_included(self):
        cfg = _all_disabled(
            record=FeatureGroupConfig(
                lags=[10], location_lags=[10], delta=True, enabled=True
            ),
        )
        cols = _without_conference(cfg)
        assert "record_L10_delta" in cols
        assert "record_L10_at_location_delta" in cols

    def test_momentum_pair_replaces_source_deltas(self):
        cfg = _record_only(lags=[5, 10])
        pair = MomentumPairConfig(feature="record", short=5, long=10)
        cols = _without_conference(cfg, momentum_pairs=[pair])
        assert "record_L5_delta" not in cols
        assert "record_L10_delta" not in cols
        assert "record_momentum_L5_L10_delta" in cols

    def test_home_and_road_raw_columns(self):
        cfg = _all_disabled(
            home_and_road=FeatureGroupConfig(enabled=True, delta=False),
        )
        cols = _without_conference(cfg)
        assert "days_at_home" in cols
        assert "days_on_road" in cols

    def test_home_and_road_delta_column(self):
        cfg = _all_disabled(
            home_and_road=FeatureGroupConfig(enabled=True, delta=True),
        )
        cols = _without_conference(cfg)
        assert "days_at_home_delta" in cols
        assert "days_at_home" not in cols


# ── create_delta_features resilience ────────────────────────────────────────


class TestCreateDeltaResilience:
    def _make_df(self, columns: list[str]) -> pd.DataFrame:
        return pd.DataFrame({col: [1.0, 2.0] for col in columns})

    def _features_config_with_record_lags(self, lags: list[int]) -> FeaturesMapConfig:
        return _all_disabled(
            record=FeatureGroupConfig(lags=lags, delta=True, enabled=True),
        )

    def test_no_crash_on_missing_columns(self):
        """create_delta_features should not raise when HT/VT columns are absent."""
        # Config says lags=[10, 82] but DataFrame only has lag 10 columns
        cfg = self._features_config_with_record_lags([10, 82])
        df = self._make_df(["record_L10_HT", "record_L10_VT"])
        result = create_delta_features(df, cfg)
        # lag 10 delta should be created; lag 82 silently skipped
        assert "record_L10_delta" in result.columns
        assert "record_L82_delta" not in result.columns

    def test_creates_delta_for_present_columns(self):
        """Deltas are correctly computed for columns that ARE present."""
        cfg = self._features_config_with_record_lags([10])
        df = self._make_df(["record_L10_HT", "record_L10_VT"])
        df["record_L10_HT"] = [0.6, 0.5]
        df["record_L10_VT"] = [0.4, 0.7]
        result = create_delta_features(df, cfg)
        assert list(result["record_L10_delta"]) == pytest.approx([0.2, -0.2])

    def test_drops_ht_vt_after_delta(self):
        cfg = self._features_config_with_record_lags([10])
        df = self._make_df(["record_L10_HT", "record_L10_VT"])
        result = create_delta_features(df, cfg)
        assert "record_L10_HT" not in result.columns
        assert "record_L10_VT" not in result.columns

    def test_sos_adj_record_delta_always_drops_source_columns(self):
        """create_delta_features drops HT/VT when delta=True for sos_adj_record."""
        cfg = _all_disabled(
            sos_adj_record=FeatureGroupConfig(enabled=True, lags=[13], delta=True),
        )
        df = pd.DataFrame({
            "sos_adj_record_L13_HT": [0.6, 0.5],
            "sos_adj_record_L13_VT": [0.4, 0.7],
        })
        result = create_delta_features(df, cfg)
        assert "sos_adj_record_L13_delta" in result.columns
        assert "sos_adj_record_L13_HT" not in result.columns
        assert "sos_adj_record_L13_VT" not in result.columns

    def test_sos_adj_record_location_delta_creation(self):
        """create_delta_features creates sos_adj_record location deltas."""
        cfg = _all_disabled(
            sos_adj_record=FeatureGroupConfig( enabled=True, lags=[13], location_lags=[10], delta=True ),
        )
        df = pd.DataFrame({
            "sos_adj_record_L13_HT": [0.6, 0.5],
            "sos_adj_record_L13_VT": [0.4, 0.7],
            "sos_adj_record_L10_HT_at_home": [0.7, 0.55],
            "sos_adj_record_L10_VT_on_road": [0.3, 0.6],
        })
        result = create_delta_features(df, cfg)
        assert "sos_adj_record_L13_delta" in result.columns
        assert "sos_adj_record_L10_at_location_delta" in result.columns
        assert list(result["sos_adj_record_L10_at_location_delta"]) == pytest.approx([0.4, -0.05])


# ── winning_incentive ────────────────────────────────────────────────────────


def _clinch_frame(rows: list[tuple[int, int, int]], suffix: str) -> pd.DataFrame:
    """One frame per side, from (eliminated, clinched_berth, clinched_seed) rows."""
    names = ("eliminated_from_playoffs", "clinched_playoff_berth", "clinched_final_seed")
    return pd.DataFrame(
        {
            f"{name}_{suffix}": [bool(row[i]) for row in rows]
            for i, name in enumerate(names)
        }
    )


class TestWinningIncentive:
    """The three clinch booleans are one ordinal spelled across three columns.

    Only four of the eight combinations are producible — compute_playoff_flags
    evaluates clinched_final_seed only when clinched_playoff_berth is true, and
    an eliminated team has clinched neither — so the mapping is exhaustive by
    construction and anything else is a broken builder, not a new case.
    """

    REACHABLE = [
        ((0, 0, 0), "fighting", 1),
        ((1, 0, 0), "eliminated", -1),
        ((0, 1, 0), "fighting for seed", 1),
        ((0, 1, 1), "resting for playoffs", 0),
    ]

    @pytest.mark.parametrize("suffix", ["HT", "VT"])
    def test_every_reachable_combination_maps(self, suffix):
        combos = [combo for combo, _, _ in self.REACHABLE]
        result = create_winning_incentive_features(_clinch_frame(combos, suffix))
        assert list(result[f"winning_incentive_label_{suffix}"]) == [
            label for _, label, _ in self.REACHABLE
        ]
        assert list(result[f"winning_incentive_{suffix}"]) == [
            value for _, _, value in self.REACHABLE
        ]

    def test_the_two_fighting_states_share_a_value(self):
        """Deliberate, not an oversight: both mean the starters are playing."""
        assert (
            WINNING_INCENTIVE_VALUES["fighting"]
            == WINNING_INCENTIVE_VALUES["fighting for seed"]
        )

    def test_unreachable_combination_raises(self):
        """Eliminated *and* clinched cannot both be true. Never default silently."""
        df = _clinch_frame([(1, 1, 0)], "HT")
        with pytest.raises(ValueError, match="Unreachable playoff-flag combination"):
            create_winning_incentive_features(df)

    def test_source_columns_are_dropped(self):
        """They stopped being feature groups, so nothing downstream drops them."""
        df = _clinch_frame([(0, 0, 0)], "HT")
        result = create_winning_incentive_features(df)
        assert not [c for c in result.columns if c.startswith("clinched_")]
        assert "eliminated_from_playoffs_HT" not in result.columns

    def test_missing_source_columns_pass_through(self):
        """A prediction frame built without the playoff table must not crash."""
        df = pd.DataFrame({"record_L10_HT": [0.5]})
        result = create_winning_incentive_features(df)
        assert list(result.columns) == ["record_L10_HT"]

    def test_delta_reads_home_minus_away(self):
        """Not inverted: more incentive is better for the team holding it."""
        df = pd.concat(
            [_clinch_frame([(0, 1, 1)], "HT"), _clinch_frame([(0, 0, 0)], "VT")],
            axis=1,
        )
        df = create_winning_incentive_features(df)
        cfg = _all_disabled(winning_incentive=FeatureGroupConfig(enabled=True, delta=True))
        result = create_delta_features(df, cfg)
        # Home team is resting (0), visitor is still fighting (1) → 0 - 1.
        assert list(result["winning_incentive_delta"]) == [-1]
        assert "winning_incentive_HT" not in result.columns

    def test_label_columns_survive_delta_creation(self):
        """The labels are for EDA; no group names them, so nothing drops them."""
        df = pd.concat(
            [_clinch_frame([(1, 0, 0)], "HT"), _clinch_frame([(0, 0, 0)], "VT")],
            axis=1,
        )
        df = create_winning_incentive_features(df)
        cfg = _all_disabled(winning_incentive=FeatureGroupConfig(enabled=True, delta=True))
        result = create_delta_features(df, cfg)
        assert result["winning_incentive_label_HT"].tolist() == ["eliminated"]
        assert result["winning_incentive_label_VT"].tolist() == ["fighting"]

    def test_labels_are_not_resolved_as_features(self):
        cfg = _all_disabled(winning_incentive=FeatureGroupConfig(enabled=True, delta=True))
        assert _without_conference(cfg) == ["winning_incentive_delta"]


# ── delta orientation ────────────────────────────────────────────────────────


class TestDeltaOrientation:
    """Every delta is signed so that a positive value favours the home team.

    For most groups that is HT - VT, because more of the quantity is better.
    For the groups in INVERTED_DELTA_GROUPS the quantity is lower-is-better —
    travelling farther and playing more back-to-backs both hurt — so the
    operands swap and the sign still points the same way.

    This is what src/ml/llm/serialization.py::SYSTEM_PROMPT tells the model
    ("Positive values favour the home team") and what a SHAP sign is read
    against, so it is worth pinning rather than leaving to a code comment.
    """

    def test_higher_is_better_group_reads_home_minus_away(self):
        cfg = _all_disabled(
            rested_days=FeatureGroupConfig(enabled=True, delta=True),
        )
        df = pd.DataFrame({"rested_days_HT": [3.0, 1.0], "rested_days_VT": [1.0, 3.0]})
        result = create_delta_features(df, cfg)
        # Home rested 3 days to the visitor's 1 → home is better off → positive.
        assert list(result["rested_days_delta"]) == pytest.approx([2.0, -2.0])

    def test_distance_reads_away_minus_home(self):
        cfg = _all_disabled(
            distance=FeatureGroupConfig(enabled=True, lags=[1], delta=True),
        )
        df = pd.DataFrame({"distance_L1_HT": [0.0, 900.0], "distance_L1_VT": [900.0, 0.0]})
        result = create_delta_features(df, cfg)
        # Visitor flew 900 miles, home team did not travel → favours home →
        # positive. The naive HT - VT would call that -900.
        assert list(result["distance_L1_delta"]) == pytest.approx([900.0, -900.0])

    def test_back_to_back_reads_away_minus_home(self):
        cfg = _all_disabled(
            back_to_back=FeatureGroupConfig(enabled=True, delta=True),
        )
        df = pd.DataFrame({"back_to_back_HT": [0, 2], "back_to_back_VT": [1, 0]})
        result = create_delta_features(df, cfg)
        # Visitor on a back-to-back, home team rested → favours home → positive.
        assert list(result["back_to_back_delta"]) == pytest.approx([1.0, -2.0])


# ── selection_mode default ───────────────────────────────────────────────────


class TestSelectionModeDefault:
    def test_default_is_exclusion(self):
        """FeatureEngineeringConfig with no selection_mode field defaults to 'exclusion'."""
        cfg = FeatureEngineeringConfig()
        assert cfg.selection_mode == "exclusion"

    def test_defaults_yaml_has_inclusion(self):
        """_defaults.yaml sets selection_mode to 'inclusion'."""
        config = load_experiment_config(TRAIN_DEFAULTS_CONFIG_PATH)
        assert config.feature_engineering.selection_mode == "inclusion"
