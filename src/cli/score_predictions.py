"""Score emitted predictions against games that have been played."""

import argparse

from src.monitoring.scoring import log_scorecard_to_mlflow, score_predictions
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join predictions to played games and report live accuracy."
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip logging the scorecard to MLflow.",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")
    scorecard = score_predictions()

    if scorecard.empty:
        return

    logger.info(f"\n{scorecard.to_string(index=False)}")
    if not args.no_mlflow:
        log_scorecard_to_mlflow(scorecard)


if __name__ == "__main__":
    main()
