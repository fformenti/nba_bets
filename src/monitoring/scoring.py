"""Score emitted predictions against games that have since been played.

Closes the loop: ``predict-upcoming`` writes predictions, games get played,
``append-game-results`` folds them into history, and this module joins the two
to answer "how is the model actually doing?".

**Outcomes come from the ingested history, not the results JSON directory.**
Those JSON files are transient — ``append-game-results`` consumes them — whereas
``games_updated_history.csv`` is the durable record of every played game and
already carries ``winner``. Joining against history makes scoring independent of
pipeline ordering: it can be re-run at any time and always reproduces the same
numbers. The results directory is only a fallback, for games fetched but not yet
appended.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config.paths import (
    INGESTED_GAMES_UPDATED_HISTORY_PATH,
    PREDICTION_SCORECARD_PATH,
    PREDICTION_SCORED_GAMES_PATH,
    UPCOMING_GAMES_PREDICTIONS_PATH,
    UPCOMING_GAMES_RESULTS_DIR,
)
from src.etl.utils.common import read_json
from src.ml.evaluation.metrics import (
    compute_brier_score,
    compute_classification_metrics,
    compute_ece,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Rolling window (in games, most recent first) for the trailing-form metric.
ROLLING_WINDOW = 100

# Identifies one prediction. Older prediction files predate conference_filter,
# so it is matched only when present.
PREDICTION_KEY = ["gameId", "conference_filter"]

# Whichever of these exists is used to order games for the trailing window.
DATE_COLUMNS = ["gameDate", "gameDateOnlyStr"]


def _date_column(df: pd.DataFrame) -> Optional[str]:
    return next((col for col in DATE_COLUMNS if col in df.columns), None)


OUTCOME_COLUMNS = ["gameId", "winner", "hometeamId"]


def _to_home_win(outcomes: pd.DataFrame) -> pd.Series:
    """Convert the stored ``winner`` (a *teamId*) into a 0/1 home-win flag.

    The models predict 1 = home win, but every outcome source records the
    winning team's id. Comparing the two directly would silently score close to
    zero, so the conversion happens here, once, for both sources.
    """
    return (outcomes["winner"] == outcomes["hometeamId"]).astype(int)


def load_outcomes(
    historical_path: Path = INGESTED_GAMES_UPDATED_HISTORY_PATH,
    results_dir: Path = UPCOMING_GAMES_RESULTS_DIR,
) -> pd.DataFrame:
    """Every known game outcome as ``gameId, actual_home_win``.

    History is authoritative; the results directory tops up games that have been
    fetched but not yet appended.
    """
    frames = []

    if historical_path.exists():
        history = pd.read_csv(
            historical_path, usecols=OUTCOME_COLUMNS, low_memory=False
        )
        frames.append(history)
        logger.info(f"Loaded {len(history)} outcomes from {historical_path.name}")
    else:
        logger.warning(f"No historical games at {historical_path}")

    if results_dir.exists():
        pending = [
            {column: payload[column] for column in OUTCOME_COLUMNS}
            for payload in (read_json(p) for p in sorted(results_dir.glob("*.json")))
            if all(payload.get(column) is not None for column in OUTCOME_COLUMNS)
        ]
        if pending:
            frames.append(pd.DataFrame(pending))
            logger.info(f"Loaded {len(pending)} outcomes not yet appended to history")

    if not frames:
        return pd.DataFrame(columns=["gameId", "actual_home_win"])

    outcomes = pd.concat(frames, ignore_index=True).dropna(subset=OUTCOME_COLUMNS)
    outcomes["gameId"] = pd.to_numeric(outcomes["gameId"], errors="coerce").astype("Int64")
    outcomes = outcomes.dropna(subset=["gameId"])
    outcomes["actual_home_win"] = _to_home_win(outcomes)

    # History wins on conflict, since it is the settled record.
    outcomes = outcomes.drop_duplicates(subset=["gameId"], keep="first")
    return outcomes[["gameId", "actual_home_win"]]


def join_predictions_to_outcomes(
    predictions: pd.DataFrame, outcomes: pd.DataFrame
) -> pd.DataFrame:
    """Inner-join predictions to outcomes, keeping only games actually played."""
    predictions = predictions.copy()
    predictions["gameId"] = pd.to_numeric(
        predictions["gameId"], errors="coerce"
    ).astype("Int64")

    # The prediction file carries its own `winner` placeholder column (always 0
    # for unplayed games); drop it so it cannot shadow the real outcome.
    predictions = predictions.drop(columns=["winner"], errors="ignore")

    scored = predictions.merge(outcomes, on="gameId", how="inner")
    scored["correct"] = (
        scored["prediction"].astype(int) == scored["actual_home_win"]
    ).astype(int)

    unplayed = len(predictions) - len(scored)
    if unplayed:
        logger.info(f"{unplayed} prediction(s) have no outcome yet; excluded")
    return scored


def _metrics_for(group: pd.DataFrame) -> dict:
    """Accuracy plus probabilistic metrics for one slice of scored games."""
    y_true = group["actual_home_win"].astype(int)
    y_pred = group["prediction"].astype(int).to_numpy()

    metrics = {"n_games": len(group)}
    metrics.update(compute_classification_metrics(y_true, y_pred))

    if "home_win_probability" in group.columns:
        proba = pd.to_numeric(group["home_win_probability"], errors="coerce")
        if proba.notna().all():
            values = proba.to_numpy()
            metrics["brier_score"] = compute_brier_score(y_true.to_numpy(), values)
            metrics["ece"] = compute_ece(y_true.to_numpy(), values)

    return metrics


def _breakdowns(scored: pd.DataFrame) -> list[dict]:
    """Overall, trailing-window, and per-slice metric rows."""
    rows = [{"scope": "overall", "group": "all", **_metrics_for(scored)}]

    date_column = _date_column(scored)
    if date_column:
        recent = scored.sort_values(date_column).tail(ROLLING_WINDOW)
        rows.append(
            {
                "scope": f"last_{ROLLING_WINDOW}",
                "group": "all",
                **_metrics_for(recent),
            }
        )

    for column, scope in [("season", "season"), ("conference_filter", "conference")]:
        if column not in scored.columns:
            continue
        for value, group in scored.groupby(column):
            rows.append({"scope": scope, "group": str(value), **_metrics_for(group)})

    return rows


def score_predictions(
    predictions_path: Path = UPCOMING_GAMES_PREDICTIONS_PATH,
    historical_path: Path = INGESTED_GAMES_UPDATED_HISTORY_PATH,
    results_dir: Path = UPCOMING_GAMES_RESULTS_DIR,
    scorecard_path: Optional[Path] = PREDICTION_SCORECARD_PATH,
    scored_games_path: Optional[Path] = PREDICTION_SCORED_GAMES_PATH,
) -> pd.DataFrame:
    """Score every prediction whose game has been played.

    Writes a per-game detail CSV and a metrics scorecard, and returns the
    scorecard. Re-running is safe and idempotent — both files are rewritten from
    scratch off the durable inputs.
    """
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"No predictions at {predictions_path}. Run `make predict-upcoming` first."
        )

    predictions = pd.read_csv(predictions_path)
    logger.info(f"Loaded {len(predictions)} prediction(s) from {predictions_path.name}")

    key = [col for col in PREDICTION_KEY if col in predictions.columns]
    duplicates = predictions.duplicated(subset=key).sum() if key else 0
    if duplicates:
        logger.warning(
            f"{duplicates} duplicate prediction row(s) found — metrics will "
            "double-count. Predictions written before the de-duplication fix "
            "need cleaning."
        )

    outcomes = load_outcomes(historical_path, results_dir)
    if outcomes.empty:
        raise ValueError(
            "No game outcomes available. Run `make fetch-game-results` and "
            "`make append-game-results`, or drop files into the manual results "
            "directory and re-run with SOURCE=placeholder."
        )

    scored = join_predictions_to_outcomes(predictions, outcomes)
    if scored.empty:
        logger.warning("No predicted games have been played yet; nothing to score.")
        return pd.DataFrame()

    scorecard = pd.DataFrame(_breakdowns(scored))

    if scored_games_path is not None:
        scored_games_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(scored_games_path, index=False)
        logger.info(f"Wrote {len(scored)} scored game(s) to {scored_games_path}")

    if scorecard_path is not None:
        scorecard_path.parent.mkdir(parents=True, exist_ok=True)
        scorecard.to_csv(scorecard_path, index=False)
        logger.info(f"Wrote scorecard to {scorecard_path}")

    overall = scorecard.iloc[0]
    logger.info(
        f"Live accuracy: {overall['accuracy']:.1%} over {int(overall['n_games'])} played game(s)"
    )
    return scorecard


def log_scorecard_to_mlflow(
    scorecard: pd.DataFrame,
    experiment_name: str = "nba_bets_monitoring",
    tracking_uri: str = "sqlite:///mlflow.db",
) -> None:
    """Log the scorecard to MLflow so live accuracy sits beside backtest accuracy."""
    from src.ml.tracking.mlflow_tracker import MLflowTracker

    if scorecard.empty:
        logger.warning("Empty scorecard; nothing logged to MLflow.")
        return

    with MLflowTracker(
        experiment_name=experiment_name,
        run_name=f"scorecard-{pd.Timestamp.now():%Y-%m-%d_%H.%M.%S}",
        tracking_uri=tracking_uri,
        log_model=False,
    ) as tracker:
        for row in scorecard.to_dict(orient="records"):
            prefix = f"{row['scope']}_{row['group']}".replace(" ", "_").replace("/", "-")
            tracker.log_metrics(
                {
                    f"{prefix}_{key}": value
                    for key, value in row.items()
                    if key not in ("scope", "group")
                    and isinstance(value, (int, float, np.floating))
                }
            )
        tracker.set_tags({"stage": "monitoring"})
    logger.info(f"Logged scorecard to MLflow experiment '{experiment_name}'")
