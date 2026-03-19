import re
from pathlib import Path
from typing import Optional

from src.config.paths import DEFAULT_TRAIN_CLASSIFIER_CONFIG_PATH, PROJECT_ROOT
from src.ml.config.loader import load_experiment_config
from src.ml.tracking.mlflow_tracker import MLflowTracker
from src.ml.training.experiment import generate_run_name, train_single_model
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)
setup_logging(level="INFO")


def main(
    config_path: Optional[Path] = None,
):
    setup_logging(level="INFO")

    resolved = Path(config_path) if config_path else DEFAULT_TRAIN_CLASSIFIER_CONFIG_PATH
    if not resolved.exists():
        raise FileNotFoundError(
            f"Training config not found: {resolved}. "
            "Pass config_path= or --config with a valid YAML (e.g. configs/train/train_same.yaml)."
        )
    config_path = resolved
    config = load_experiment_config(config_path)

    conference_filter = config.filters.conference_filter
    run_name = generate_run_name(config_path=config_path, include_timestamp=True)

    with MLflowTracker(
        experiment_name=config.mlflow.experiment_name,
        run_name=run_name,
        tracking_uri=config.mlflow.tracking_uri,
    ) as tracker:
        tracker.log_config(config.model_dump())
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
        logger.info(f"{conference_filter}: {best_model} - Accuracy: {best_acc:.4f}")

    _update_predict_config(
        conference_filter, result.get("run_model_uri") if result else None
    )

    logger.info("Training complete!")
    return result


def _update_predict_config(conference_filter: str, model_uri: Optional[str]) -> None:
    """Rewrite the model URI for this conference_filter in predict_classifier.yaml."""
    predict_config_path = (
        PROJECT_ROOT / "configs" / "predict" / "predict_classifier.yaml"
    )
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
    content = re.sub(
        rf'^(\s+{conference_filter}:\s+").*(")',
        rf"\g<1>{model_uri}\2",
        content,
        flags=re.MULTILINE,
    )

    predict_config_path.write_text(content)
    logger.info(f"Updated {predict_config_path} with new model URI:")
    logger.info(f"  {conference_filter}: {model_uri}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train classification model")
    parser.add_argument("--config", type=Path, help="Path to configuration YAML file")
    args = parser.parse_args()

    main(config_path=args.config)
