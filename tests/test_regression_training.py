"""A margin target trains regressors through the same entry point as win_bool."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import is_classifier, is_regressor
from sklearn.preprocessing import StandardScaler

from src.config.paths import CONFIGS_TRAIN_DIR
from src.ml.config.loader import load_experiment_config
from src.ml.training.classifier import train_classifier
from src.ml.training.model_factory import create_model
from src.ml.training.regressor import winner_accuracy
from src.ml.training.runners import train_model_with_config

FAMILIES = ["random_forest", "gradient_boosting", "xgboost", "lgbm", "neural_network"]
XGBOOST_MARGIN = CONFIGS_TRAIN_DIR / "xgboost_margin.yaml"


@pytest.mark.parametrize("family", FAMILIES)
def test_each_family_has_a_regressor_and_a_classifier(family):
    regressor, _ = create_model(family, {}, task_type="regression")
    classifier, _ = create_model(family, {}, task_type="classification")

    assert is_regressor(regressor)
    assert is_classifier(classifier)
    # Class balancing has no meaning for a margin, so it is never set.
    for knob in ("class_weight", "scale_pos_weight"):
        assert regressor.get_params().get(knob) is None


def test_winner_accuracy_scores_the_sign_of_the_margin():
    y_true = [12, -3, 7, -20]
    y_pred = [1.5, 2.0, 9.0, -0.5]  # right, wrong, right, right
    assert winner_accuracy(y_true, y_pred) == 0.75


def test_a_regressor_trains_and_reports_mae():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    y = pd.Series(8 * X["a"] + rng.normal(scale=2, size=400))

    pipeline, trainer, results = train_model_with_config(
        model_name="xgboost",
        model_config={"xgboost": {"n_estimators": 50}},
        X_train=X[:300],
        y_train=y[:300],
        X_val=X[300:],
        y_val=y[300:],
        preprocessor=StandardScaler(),
        random_state=0,
        task_type="regression",
    )

    assert is_regressor(pipeline.named_steps["model"])
    assert results["val"]["val_mae"] < y[300:].abs().mean()


def test_the_margin_config_is_a_regression_without_feature_selection():
    config = load_experiment_config(XGBOOST_MARGIN)
    assert config.data.target_column == "margin"
    assert config.data.task_type == "regression"
    assert config.feature_selection.enabled is False
    assert config.model.xgboost["hyperparameter_tuning"]["scoring"] == "neg_mean_absolute_error"


def test_a_regression_run_refuses_to_promote():
    # Prediction serves win probabilities; a margin regressor has none.
    with pytest.raises(ValueError, match="Refusing --promote"):
        train_classifier(config_path=XGBOOST_MARGIN, promote=True)
