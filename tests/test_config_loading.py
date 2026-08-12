"""Tests for the _include-based config loading system."""

import pytest

from src.config.paths import CONFIGS_TRAIN_DIR, PROJECT_ROOT
from src.ml.config.loader import (
    load_experiment_config,
    load_features_config,
    _deep_merge,
)

CONFIGS_DIR = PROJECT_ROOT / "configs"
FEATURES_YAML = CONFIGS_DIR / "features.yaml"
# The deployed model. The same/different conference pair it replaced lost a
# paired backtest — see docs/CONFERENCE_SPLIT.md.
XGBOOST = CONFIGS_TRAIN_DIR / "xgboost.yaml"
# Same features and splits as XGBOOST, four model families instead of one.
ALL_MODELS = CONFIGS_TRAIN_DIR / "all_models.yaml"
# An ExperimentConfig too, but it deliberately diverges: it enables `record` and
# `streak` and keeps home_and_road raw.
LLM_FEATURES = CONFIGS_TRAIN_DIR / "llm_features.yaml"

# The two that must resolve to an identical feature set.
SKLEARN_TRAIN_CONFIGS = [XGBOOST, ALL_MODELS]
# Every runnable train config, for assertions that hold regardless of feature set.
ALL_TRAIN_CONFIGS = SKLEARN_TRAIN_CONFIGS + [LLM_FEATURES]


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

    @pytest.mark.parametrize(
        "config_path", SKLEARN_TRAIN_CONFIGS, ids=lambda p: p.stem
    )
    def test_defaults_override_features_yaml(self, config_path):
        """_defaults.yaml wins over features.yaml through the include chain."""
        fe = load_experiment_config(config_path).feature_engineering
        # features.yaml enables `record`; _defaults.yaml turns it off for training
        assert fe.features.record.enabled is False
        # features.yaml builds L82; the model stops at L55
        assert fe.features.norm_point_differential.enabled is True
        assert fe.features.norm_point_differential.lags == [1, 3, 5, 8, 13, 21, 34, 55]

    @pytest.mark.parametrize("config_path", ALL_TRAIN_CONFIGS, ids=lambda p: p.stem)
    def test_opt_in_groups_disabled_by_default(self, config_path):
        """Groups no experiment enables stay off after the merge."""
        features = load_experiment_config(config_path).feature_engineering.features
        assert features.sos.enabled is False
        assert features.last_season_record.enabled is False
        assert features.neutral_court.enabled is False

    @pytest.mark.parametrize(
        "config_path", SKLEARN_TRAIN_CONFIGS, ids=lambda p: p.stem
    )
    def test_streak_off_for_sklearn(self, config_path):
        """`streak` is opt-in; only llm_features.yaml takes it."""
        features = load_experiment_config(config_path).feature_engineering.features
        assert features.streak.enabled is False

    @pytest.mark.parametrize("config_path", ALL_TRAIN_CONFIGS, ids=lambda p: p.stem)
    def test_sos_adj_alpha_from_features_yaml(self, config_path):
        """sos_adj_alpha merges from features.yaml; delta settings come from _defaults.yaml."""
        features = load_features_config(FEATURES_YAML)
        config = load_experiment_config(config_path)
        assert config.feature_engineering.sos_adj_alpha == features.sos_adj_alpha
        assert config.feature_engineering.features.sos.delta is False

    @pytest.mark.parametrize("config_path", ALL_TRAIN_CONFIGS, ids=lambda p: p.stem)
    def test_include_tentative_inherited_from_defaults(self, config_path):
        config = load_experiment_config(config_path)
        assert config.feature_selection.include_tentative is True

    def test_delta_flags_come_from_defaults_yaml(self):
        """features.yaml declares no delta; _defaults.yaml supplies it for every group."""
        features = load_experiment_config(XGBOOST).feature_engineering.features
        assert features.sos.delta is False
        assert features.neutral_court.delta is False
        assert features.distance.delta is True
        assert features.streak.delta is True

    def test_model_hyperparameter_defaults_from_defaults_yaml(self):
        """Classifier blocks inherit hyperparameter_tuning from _defaults.yaml."""
        config = load_experiment_config(XGBOOST)
        assert config.model.gradient_boosting["hyperparameter_tuning"]["cv_folds"] == 5
        assert config.model.random_forest["hyperparameter_tuning"]["cv_folds"] == 5

    def test_xgboost_learning_rate_grid(self):
        config = load_experiment_config(XGBOOST)
        lr = config.model.xgboost["hyperparameter_tuning"]["param_grid"][
            "learning_rate"
        ]
        assert lr == [0.01, 0.05, 0.1]

    def test_xgboost_feature_set_unchanged_by_defaults_move(self):
        """Pins the feature set that moved out of xgb_all.yaml into _defaults.yaml.

        These five values were spelled out in the experiment config before the
        configs/ cleanup; they are now inherited. If a future edit to
        _defaults.yaml changes any of them, it changes what the deployed model
        trains on, and that should be a deliberate act rather than a side effect.
        """
        config = load_experiment_config(XGBOOST)
        features = config.feature_engineering.features
        assert features.norm_point_differential.lags == [1, 3, 5, 8, 13, 21, 34, 55]
        assert features.sos_adj_record.lags == []
        assert features.sos_adj_record.location_lags == [5, 10, 20, 41]
        assert features.record.enabled is False
        # A sum, not a difference — days at home and days on road both favour the
        # home team, so the two columns collapse into one. See _defaults.yaml.
        assert features.home_and_road.delta is True
        assert config.feature_selection.include_tentative is True

    def test_all_models_differs_from_xgboost_only_in_model_choice(self):
        """The sweep config must not drift into a different feature set."""
        xgb = load_experiment_config(XGBOOST)
        sweep = load_experiment_config(ALL_MODELS)
        assert (
            sweep.feature_engineering.model_dump()
            == xgb.feature_engineering.model_dump()
        )
        assert sweep.splitting.model_dump() == xgb.splitting.model_dump()
        assert sweep.filters.model_dump() == xgb.filters.model_dump()
        assert xgb.model.train_models == ["xgboost"]
        assert sweep.model.train_models == [
            "random_forest",
            "gradient_boosting",
            "xgboost",
            "neural_network",
        ]

    def test_llm_features_diverges_where_intended(self):
        """llm_features.yaml overrides exactly three groups; the rest is inherited."""
        llm = load_experiment_config(LLM_FEATURES).feature_engineering.features
        xgb = load_experiment_config(XGBOOST).feature_engineering.features
        assert llm.record.enabled is True and xgb.record.enabled is False
        assert llm.streak.enabled is True and xgb.streak.enabled is False
        assert llm.home_and_road.delta is False and xgb.home_and_road.delta is True
        # Everything else tracks the shared base.
        assert llm.norm_point_differential.lags == xgb.norm_point_differential.lags
        assert llm.sos_adj_record.lags == xgb.sos_adj_record.lags
        assert llm.distance.model_dump() == xgb.distance.model_dump()

    @pytest.mark.parametrize("config_path", ALL_TRAIN_CONFIGS, ids=lambda p: p.stem)
    def test_model_dump_roundtrip(self, config_path):
        """Resolved config should produce a valid model_dump."""
        config = load_experiment_config(config_path)
        dumped = config.model_dump()
        assert isinstance(dumped, dict)
        assert "feature_engineering" in dumped
        assert "model" in dumped
        assert "mlflow" in dumped
