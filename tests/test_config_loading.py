"""Tests for the _include-based config loading system."""

import pytest
from pathlib import Path

from src.ml.config.loader import (
    load_experiment_config,
    load_features_config,
    _deep_merge,
)

CONFIGS_DIR = Path("configs")
FEATURES_YAML = CONFIGS_DIR / "features.yaml"
TRAIN_SAME = CONFIGS_DIR / "train" / "train_same.yaml"
TRAIN_DIFFERENT = CONFIGS_DIR / "train" / "train_different.yaml"
TRAIN_ALL = CONFIGS_DIR / "train" / "train_all.yaml"

ALL_TRAIN_CONFIGS = [TRAIN_SAME, TRAIN_DIFFERENT, TRAIN_ALL]


# ── _deep_merge unit tests ──────────────────────────────────────────


class TestDeepMerge:
    def test_overlay_wins_for_scalars(self):
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_recursive_dict_merge(self):
        base = {"x": {"a": 1, "b": 2}}
        overlay = {"x": {"b": 3, "c": 4}}
        assert _deep_merge(base, overlay) == {"x": {"a": 1, "b": 3, "c": 4}}

    def test_overlay_replaces_lists(self):
        base = {"items": [1, 2, 3]}
        overlay = {"items": [4, 5]}
        assert _deep_merge(base, overlay) == {"items": [4, 5]}

    def test_base_keys_preserved(self):
        base = {"a": 1, "b": 2}
        overlay = {"b": 3}
        assert _deep_merge(base, overlay) == {"a": 1, "b": 3}

    def test_does_not_mutate_inputs(self):
        base = {"x": {"a": 1}}
        overlay = {"x": {"b": 2}}
        _deep_merge(base, overlay)
        assert base == {"x": {"a": 1}}
        assert overlay == {"x": {"b": 2}}


# ── features.yaml loading ───────────────────────────────────────────


class TestFeaturesConfig:
    def test_load_features_config(self):
        fe = load_features_config(FEATURES_YAML)
        assert fe.record_lags == [1, 3, 5, 8, 13, 21, 34, 55, 82]
        assert fe.sos_adj_alpha == 1.0

    def test_features_yaml_has_all_lag_types(self):
        fe = load_features_config(FEATURES_YAML)
        assert len(fe.record_lags) > 0
        assert len(fe.point_differential_lags) > 0
        assert len(fe.location_lags) > 0
        assert len(fe.distances_lags) > 0
        assert len(fe.sos_lags) > 0


# ── train config loading (include resolution) ───────────────────────


class TestTrainConfigLoading:
    @pytest.mark.parametrize("config_path", ALL_TRAIN_CONFIGS, ids=lambda p: p.stem)
    def test_loads_without_error(self, config_path):
        config = load_experiment_config(config_path)
        assert config is not None

    @pytest.mark.parametrize("config_path", ALL_TRAIN_CONFIGS, ids=lambda p: p.stem)
    def test_lag_shortcut_properties_mirror_feature_groups(self, config_path):
        """The flat *_lags properties are views onto the resolved feature groups."""
        fe = load_experiment_config(config_path).feature_engineering
        assert fe.record_lags == fe.features.record.lags
        assert fe.point_differential_lags == fe.features.point_differential.lags
        assert fe.location_lags == fe.features.record.location_lags
        assert fe.distances_lags == fe.features.distance.lags
        assert fe.sos_lags == fe.features.sos.lags

    @pytest.mark.parametrize("config_path", ALL_TRAIN_CONFIGS, ids=lambda p: p.stem)
    def test_experiment_overrides_win_over_includes(self, config_path):
        """Per-experiment feature blocks override both features.yaml and _defaults."""
        fe = load_experiment_config(config_path).feature_engineering
        # every train config disables `record` and enables `norm_point_differential`,
        # overriding features.yaml (record enabled) via the include chain
        assert fe.features.record.enabled is False
        assert fe.features.record.lags == []
        assert fe.features.norm_point_differential.enabled is True
        assert fe.features.norm_point_differential.lags == [1, 3, 5, 8, 13, 21, 34, 55]

    @pytest.mark.parametrize("config_path", ALL_TRAIN_CONFIGS, ids=lambda p: p.stem)
    def test_opt_in_groups_disabled_by_default(self, config_path):
        """Groups no experiment enables stay off after the merge."""
        features = load_experiment_config(config_path).feature_engineering.features
        assert features.sos.enabled is False
        assert features.streak.enabled is False
        assert features.last_season_record.enabled is False
        assert features.neutral_court.enabled is False

    @pytest.mark.parametrize("config_path", ALL_TRAIN_CONFIGS, ids=lambda p: p.stem)
    def test_sos_adj_alpha_from_features_yaml(self, config_path):
        """sos_adj_alpha merges from features.yaml; delta settings come from _defaults.yaml."""
        features = load_features_config(FEATURES_YAML)
        config = load_experiment_config(config_path)
        assert config.feature_engineering.sos_adj_alpha == features.sos_adj_alpha
        assert config.feature_engineering.features.sos.delta is False

    def test_train_same_conference_filter(self):
        config = load_experiment_config(TRAIN_SAME)
        assert config.filters.conference_filter == "same"

    def test_train_different_conference_filter(self):
        config = load_experiment_config(TRAIN_DIFFERENT)
        assert config.filters.conference_filter == "different"

    def test_train_all_conference_filter(self):
        config = load_experiment_config(TRAIN_ALL)
        assert config.filters.conference_filter == "all"

    @pytest.mark.parametrize("config_path", ALL_TRAIN_CONFIGS, ids=lambda p: p.stem)
    def test_include_tentative_inherited_from_defaults(self, config_path):
        config = load_experiment_config(config_path)
        assert config.feature_selection.include_tentative is True

    def test_delta_flags_come_from_defaults_yaml(self):
        """features.yaml declares no delta; _defaults.yaml supplies it for every group."""
        features = load_experiment_config(TRAIN_SAME).feature_engineering.features
        assert features.sos.delta is False
        assert features.neutral_court.delta is False
        assert features.distance.delta is True
        assert features.streak.delta is True

    def test_model_hyperparameter_defaults_from_defaults_yaml(self):
        """Classifier blocks inherit hyperparameter_tuning from _defaults.yaml."""
        config = load_experiment_config(TRAIN_DIFFERENT)
        assert config.model.gradient_boosting["hyperparameter_tuning"]["cv_folds"] == 5
        assert config.model.random_forest["hyperparameter_tuning"]["cv_folds"] == 5

    def test_train_all_xgboost_learning_rate_grid(self):
        config = load_experiment_config(TRAIN_ALL)
        lr = config.model.xgboost["hyperparameter_tuning"]["param_grid"][
            "learning_rate"
        ]
        assert lr == [0.01, 0.05, 0.1]

    @pytest.mark.parametrize("config_path", ALL_TRAIN_CONFIGS, ids=lambda p: p.stem)
    def test_model_dump_roundtrip(self, config_path):
        """Resolved config should produce a valid model_dump."""
        config = load_experiment_config(config_path)
        dumped = config.model_dump()
        assert isinstance(dumped, dict)
        assert "feature_engineering" in dumped
        assert "model" in dumped
        assert "mlflow" in dumped
