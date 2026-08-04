"""Build the LLM's HuggingFace dataset from the ML models' splits.

Same gameIds per split as the sklearn experiment named in the LLM config's
``data.source_experiment_config`` — that parity is what makes the two model
families comparable.
"""

import argparse
from pathlib import Path

from src.config.paths import DEFAULT_TRAIN_LLM_CONFIG_PATH
from src.ml.config.loader import load_llm_training_config
from src.ml.llm.dataset import build_llm_dataset, push_to_hub, summarize
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the LLM training dataset")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_TRAIN_LLM_CONFIG_PATH,
        help=f"LLM config YAML (default: {DEFAULT_TRAIN_LLM_CONFIG_PATH})",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Upload to the Hugging Face Hub. Without this the dataset is only summarised.",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")
    config = load_llm_training_config(args.config)
    dataset = build_llm_dataset(config)

    logger.info(f"\n{summarize(dataset).to_string(index=False)}")

    if args.push:
        push_to_hub(dataset, config.data.dataset_name, private=config.hub.private)
    else:
        logger.info("Dry run — pass --push to upload to the Hub.")


if __name__ == "__main__":
    main()
