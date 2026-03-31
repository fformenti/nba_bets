"""Tests for the _include-based config loading system."""

import pytest
from pathlib import Path

from src.ml.config.loader import (
    load_experiment_config,
    load_features_config,
    load_yaml_config,
    _deep_merge,
    _resolve_includes,
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

    def test_train_same_record_and_point_diff_lags(self):
        """train_same overrides lags on top of merged features + _defaults."""
        config = load_experiment_config(TRAIN_SAME)
        assert config.feature_engineering.record_lags == [13, 34, 55, 82]
        assert config.feature_engineering.point_differential_lags == [1, 3, 5, 8, 13, 21, 34, 55, 82]

    @pytest.mark.parametrize("config_path", [TRAIN_DIFFERENT, TRAIN_ALL], ids=lambda p: p.stem)
    def test_train_different_all_full_lag_lists(self, config_path):
        """Configs without per-experiment lag overrides keep merged defaults."""
        full = [1, 3, 5, 8, 13, 21, 34, 55, 82]
        config = load_experiment_config(config_path)
        assert config.feature_engineering.record_lags == full
        assert config.feature_engineering.point_differential_lags == full

    @pytest.mark.parametrize("config_path", [TRAIN_DIFFERENT, TRAIN_ALL], ids=lambda p: p.stem)
    def test_non_selected_groups_disabled(self, config_path):
        """Feature groups that are opt-in are disabled by default."""
        config = load_experiment_config(config_path)
        features = config.feature_engineering.features
        assert features.sos.enabled is False
        assert features.sos_adj_record.enabled is False
        assert features.streak.enabled is False
        assert features.last_season_record.enabled is False
        assert features.home_and_road.enabled is False

    def test_train_same_sos_adj_record_enabled(self):
        """train_same explicitly enables sos_adj_record with its own independent lags."""
        config = load_experiment_config(TRAIN_SAME)
        assert config.feature_engineering.features.sos_adj_record.enabled is True
        assert config.feature_engineering.features.sos_adj_record.lags == [13, 34, 55, 82]

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

    def test_train_same_include_tentative_false(self):
        config = load_experiment_config(TRAIN_SAME)
        assert config.feature_selection.include_tentative is False

    def test_train_different_include_tentative_true(self):
        config = load_experiment_config(TRAIN_DIFFERENT)
        assert config.feature_selection.include_tentative is True

    def test_features_config_has_feature_groups(self):
        config = load_experiment_config(TRAIN_SAME)
        features = config.feature_engineering.features
        assert features.record.enabled is True
        assert features.record.delta is True
        assert features.sos.delta is False
        assert features.home_and_road.delta is False
        assert features.home_and_road.enabled is False

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
