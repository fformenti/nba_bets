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
# An ExperimentConfig too, but it deliberately diverges: it enables `streak` and
# keeps home_and_road raw.
LLM_FEATURES = CONFIGS_TRAIN_DIR / "llm_features.yaml"
# Diagnostic only. Enables every group at once — including both members of every
# coupled pair — so it is exempt from test_no_coupled_pairs_co_enabled.
FEATURE_AUDIT = CONFIGS_TRAIN_DIR / "feature_audit.yaml"

# The two that must resolve to an identical feature set.
SKLEARN_TRAIN_CONFIGS = [XGBOOST, ALL_MODELS]
# Every runnable train config, for assertions that hold regardless of feature set.
ALL_TRAIN_CONFIGS = SKLEARN_TRAIN_CONFIGS + [LLM_FEATURES]

# ── The redundancy map, as an executable constraint ────────────────────────
#
# Pairs of feature groups where one is computed from the other, or where the two
# measure Pearson r > 0.95 on the 49,175 training rows. Exactly one member of
# each pair may be enabled in a shippable config.
#
# This exists because the failure it catches is invisible in any single file:
# `record` is disabled in _defaults.yaml, sos_adj_record is enabled there, and
# whether both end up live depends on a leaf config three includes away. On
# 2026-08-12 two such pairs went out co-enabled and cost 0.66pp of test accuracy.
#
# Do NOT relax a pair here to make a config pass. The decision the pair encodes —
# which member is the training-facing one — belongs in the comment at the
# disabled group in _defaults.yaml and in the redundancy map at
# .claude/skills/data-analyst/references/feature-catalog.md.
COUPLED_GROUPS = [
    (
        "point_differential",
        "norm_point_differential",
        "siblings from one pts_diff column; norm divides by a near-constant "
        "within-season mean, so r = 0.998 at every lag",
    ),
    (
        "last_season_record",
        "adjusted_last_season_record",
        "same builder chain with _apply_sos_to_season_record inserted; r = 0.9985",
    ),
    (
        "record",
        "sos_adj_record",
        "sos_adj_record_L{lag} = record_L{lag} * (sos/league_avg)^alpha; r = 0.95",
    ),
    (
        "sos",
        "sos_adj_record",
        "sos is the multiplier in that same expression",
    ),
    (
        "sos",
        "adjusted_last_season_record",
        "_apply_sos_to_season_record adjusts by each team's end-of-season sos_L82",
    ),
    (
        "sos_adj_record",
        "gds",
        "gds takes opponent quality Q from _find_largest_sos_adj_col(), i.e. "
        "sos_adj_record_L82; r = 0.946",
    ),
    (
        "record",
        "gds",
        "gds falls back to cumulative win% via _build_cum_win_pct_lookup; r = 0.967",
    ),
]


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
        # features.yaml declares no `enabled` at all — every one of them was
        # overridden here and five asserted the opposite of what shipped, so the
        # keys were removed. _defaults.yaml is the only place training
        # enablement is decided, and it turns `record` off.
        assert fe.features.record.enabled is False
        assert fe.features.norm_point_differential.enabled is True
        # The model takes every window the ETL builds, L82 included — confirmed
        # at 20/20 hits, and previously excluded by an undocumented L55 cap.
        assert fe.features.norm_point_differential.lags == [1, 3, 5, 8, 13, 21, 34, 55, 82]
        # Its raw twin stays off: r = 0.998 against the above at every lag.
        assert fe.features.point_differential.enabled is False

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
        assert features.norm_point_differential.lags == [1, 3, 5, 8, 13, 21, 34, 55, 82]
        assert features.sos_adj_record.lags == []
        assert features.sos_adj_record.location_lags == [5, 10, 20, 41]
        assert features.record.enabled is False
        # A sum, not a difference — days at home and days on road both favour the
        # home team, so the two columns collapse into one. See _defaults.yaml.
        assert features.home_and_road.delta is True
        assert config.feature_selection.include_tentative is True

    def test_only_games_behind_leader_survived_the_feature_audit(self):
        """The feature audit enabled several groups; most were reverted.

        `gds` and `point_differential` were confirmed by Boruta-SHAP and still
        turned back off, because each sat next to a feature it is derived from or
        near-duplicates (see COUPLED_GROUPS). Of the six groups exploded out of
        the old bundled `playoff_standings`, only `games_behind_leader`
        survived: no derivation edge to an enabled group, and it correlates
        -0.80 with record_L82, under the 0.95 threshold. The other five were
        rejected at 0 hits (or, for eliminated_from_playoffs/clinched_final_seed,
        never reachable before the split).

        Re-enabling any of these is a modelling decision that needs a
        head-to-head against the group it duplicates — not another Boruta run.
        """
        features = load_experiment_config(XGBOOST).feature_engineering.features
        assert features.gds.enabled is False
        assert features.point_differential.enabled is False
        assert features.games_behind_leader.enabled is True
        assert features.games_behind_leader.delta is True
        assert features.games_behind_above.enabled is False
        assert features.games_ahead_of_below.enabled is False
        assert features.clinched_playoff_berth.enabled is False
        assert features.eliminated_from_playoffs.enabled is False
        assert features.clinched_final_seed.enabled is False

    @pytest.mark.parametrize("config_path", ALL_TRAIN_CONFIGS, ids=lambda p: p.stem)
    def test_no_coupled_pairs_co_enabled(self, config_path):
        """No shippable config may enable both members of a coupled pair.

        See COUPLED_GROUPS for the map and the reasoning behind each edge.
        """
        features = load_experiment_config(config_path).feature_engineering.features
        violations = [
            f"{a} + {b} ({why})"
            for a, b, why in COUPLED_GROUPS
            if getattr(features, a).enabled and getattr(features, b).enabled
        ]
        assert not violations, (
            f"{config_path.name} enables both members of a coupled feature pair:\n  "
            + "\n  ".join(violations)
            + "\nExactly one member may be enabled. Pick the training-facing one "
            "and say why at the disabled group in _defaults.yaml."
        )

    def test_feature_audit_is_exempt_and_still_violates_the_rule(self):
        """The diagnostic config must keep enabling coupled pairs.

        Its whole job is "has Boruta ever seen this feature", which requires
        every group on at once. If this ever stops failing the coupled-pair
        check, someone has quietly turned it into a second shipping config and
        the audit no longer covers the groups _defaults.yaml disables.
        """
        features = load_experiment_config(FEATURE_AUDIT).feature_engineering.features
        co_enabled = [
            (a, b)
            for a, b, _ in COUPLED_GROUPS
            if getattr(features, a).enabled and getattr(features, b).enabled
        ]
        assert len(co_enabled) == len(COUPLED_GROUPS), (
            "feature_audit.yaml should enable every group, so every coupled pair "
            f"should be co-enabled. Missing: "
            f"{[p for p in COUPLED_GROUPS if (p[0], p[1]) not in co_enabled]}"
        )

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
        """llm_features.yaml overrides exactly two groups; the rest is inherited.

        It used to override three. `record` was dropped when the redundancy rule
        was applied uniformly: this config also enables sos_adj_record, which is
        computed from record (r = 0.95), and unlike the sklearn path there is no
        Boruta pass here to filter the duplicate out — nothing trains from this
        file, so a redundant column goes straight into the prompt.
        """
        llm = load_experiment_config(LLM_FEATURES).feature_engineering.features
        xgb = load_experiment_config(XGBOOST).feature_engineering.features
        assert llm.record.enabled is False and xgb.record.enabled is False
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
