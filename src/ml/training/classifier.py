import re
from pathlib import Path
from typing import Optional

import yaml

from src.config.paths import (
    DEFAULT_FEATURES_CONFIG_PATH,
    DEFAULT_PREDICT_CONFIG_PATH,
    DEFAULT_TRAIN_CLASSIFIER_CONFIG_PATH,
)
from src.ml.config.loader import load_experiment_config
from src.ml.config.schema import ExperimentConfig
from src.ml.tracking.mlflow_tracker import MLflowTracker
from src.ml.training.experiment import generate_run_name, train_single_model
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)
setup_logging(level="INFO")


def _log_feature_configs(
    config: ExperimentConfig, config_path: Path, tracker: MLflowTracker
) -> None:
    """Log feature-related params from the resolved config.

    Reads the merged ``ExperimentConfig``, not the raw leaf YAML. Reading the
    leaf meant a value inherited from ``_defaults.yaml`` rather than restated in
    the experiment config went unlogged — so tidying a redundant line out of a
    config silently changed what MLflow recorded about the run.
    """
    include_tentative = config.feature_selection.include_tentative
    sos_adj_alpha = config.feature_engineering.sos_adj_alpha

    tracker.log_params(
        {
            "feature_selection_include_tentative": include_tentative,
            "feature_engineering_sos_adj_alpha": sos_adj_alpha,
        }
    )

    tracker.log_dict(
        {
            "train_config_path": str(config_path),
            "features_config_path": str(DEFAULT_FEATURES_CONFIG_PATH),
            "feature_selection": config.feature_selection.model_dump(),
            "feature_engineering": {
                "sos_adj_alpha": sos_adj_alpha,
            },
        },
        artifact_file="feature_tracking/feature_config_snapshot.json",
    )


def train_classifier(config_path: Optional[Path] = None, promote: bool = False):
    """Train every model in the config and register the best with MLflow.

    Parameters
    ----------
    promote : bool, default False
        Point ``configs/predict/predict_classifier.yaml`` at the model this run
        produced — i.e. deploy it. Off by default: this used to happen on every
        run, so any experiment, smoke test or worse-scoring model silently took
        over live inference. Deploying is a decision, not a side effect of
        training. The run URI is always logged, so promoting later is a copy and
        a paste, or a re-run with ``--promote``.
    """
    setup_logging(level="INFO")

    resolved = Path(config_path) if config_path else DEFAULT_TRAIN_CLASSIFIER_CONFIG_PATH
    if not resolved.exists():
        raise FileNotFoundError(
            f"Training config not found: {resolved}. "
            "Pass config_path= or --config with a valid YAML (e.g. configs/train/xgboost.yaml)."
        )
    config_path = resolved
    config = load_experiment_config(config_path)

    run_name = generate_run_name(config_path=config_path, include_timestamp=True)

    with MLflowTracker(
        experiment_name=config.mlflow.experiment_name,
        run_name=run_name,
        tracking_uri=config.mlflow.tracking_uri,
    ) as tracker:
        tracker.log_config(config.model_dump())
        _log_feature_configs(config, config_path, tracker)
        result = train_single_model(
            config=config,
            config_path=config_path,
            tracker=tracker,
        )

    if result:
        best_model = result.get("best_model_name", "N/A")
        best_acc = (
            result.get("best_model_data", {})
            .get("test_metrics", {})
            .get("test_accuracy", 0)
        )
        logger.info(f"{best_model} - Accuracy: {best_acc:.4f}")

    run_model_uri = result.get("run_model_uri") if result else None
    if promote:
        _update_predict_config(run_model_uri)
    elif run_model_uri:
        logger.info(
            "Not deploying (pass --promote to point prediction at this run):\n  %s",
            run_model_uri,
        )

    logger.info("Training complete!")
    return result


def _update_predict_config(model_uri: Optional[str]) -> None:
    """Point predict_classifier.yaml at ``model_uri``."""
    predict_config_path = DEFAULT_PREDICT_CONFIG_PATH
    if not predict_config_path.exists():
        logger.warning(
            f"Predict config not found at {predict_config_path}, skipping URI update."
        )
        return

    if not model_uri:
        logger.warning(
            "No run URI found in training results, skipping predict config update."
        )
        return

    content = predict_config_path.read_text()
    # Everything the match consumes is captured and re-emitted. An earlier
    # version left its anchor outside the capture groups and so deleted the key
    # it was editing; the rewrite is re-parsed below because a deploy step that
    # cannot verify it deployed is how that went unnoticed.
    pattern = r'(?m)^(model_uri:\s+")([^"]*)(")'
    content, substitutions = re.subn(
        pattern, rf"\g<1>{model_uri}\g<3>", content, count=1
    )
    if not substitutions:
        logger.warning(
            f"No 'model_uri' entry found in {predict_config_path}; leaving it "
            f"untouched. Set it by hand: {model_uri}"
        )
        return

    try:
        deployed = (yaml.safe_load(content) or {}).get("model_uri")
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Promoting would leave {predict_config_path} unparseable ({exc}); "
            f"refusing to write. Set it by hand: {model_uri}"
        ) from exc

    if deployed != model_uri:
        raise ValueError(
            f"Promotion did not take effect: the rewritten {predict_config_path} "
            f"reads {deployed!r} rather than {model_uri!r}. Refusing to write."
        )

    predict_config_path.write_text(content)
    logger.info(f"Updated {predict_config_path} with new model URI: {model_uri}")
