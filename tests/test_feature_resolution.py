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
    create_delta_features,
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
