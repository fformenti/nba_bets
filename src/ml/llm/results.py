"""Turn an LLM inspection frame into the shared result files.

``inspect_llm`` scores an adapter game by game; this writes those games into
``outputs/runs/<run_name>/`` in the schema every model family shares, so the
adapter's predictions plot next to the sklearn models' on the same gameIds.
The LLM-only columns (raw text, first-token probabilities) stay in
``llm/inspection_<split>.csv`` beside them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.config.paths import RUNS_DIR, RunPaths
from src.ml.llm.diagnostics import _home_won, read_inspection_csv, recalibrate, summarize
from src.ml.results import (
    RunInfo,
    build_predictions,
    git_commit,
    read_run_info,
    write_predictions,
    write_run_info,
)
from src.ml.results.store import write_json
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _series(frame: pd.DataFrame, run_id: str, split: str, calibration: str) -> pd.DataFrame:
    """One calibration of ``frame`` in the shared schema.

    The hard call is ``p_home > 0.5`` - the scoreboard's ``accuracy_p_home`` -
    rather than the greedy answer, so Platt (which moves only ``p_home``) changes
    it as it changes the scoreboard. The greedy answer stays in the inspection
    CSV as ``parsed``.
    """
    p_home = frame["p_home"].astype(float)
    pred_margin = None
    if calibration == "none" and {"expected_margin", "margin_scale"} <= set(frame.columns):
        # Calibration moves the probability only; the margin belongs to the raw model.
        pred_margin = frame["expected_margin"] * frame["margin_scale"]
    return build_predictions(
        run_id=run_id,
        model=run_id,
        split=split,
        calibration=calibration,
        game_ids=frame["game_id"],
        y_true=_home_won(frame).astype(int),
        y_pred=(p_home > 0.5).astype(int),
        p_home=p_home,
        pred_margin=pred_margin,
        true_margin=frame["margin"] if "margin" in frame.columns else None,
    )


def inspection_predictions(
    frame: pd.DataFrame,
    run_id: str,
    split: str,
    calibration_frame: Optional[pd.DataFrame],
    seed: int,
) -> pd.DataFrame:
    """Raw and Platt-recalibrated series for one split.

    Platt is cross-fitted on validation and, for any other split, fitted on the
    same run's validation - what the scoreboard has always done. Without a
    validation frame to fit on, only the raw series is returned.
    """
    series = [_series(frame, run_id, split, "none")]
    if split == "validation":
        series.append(_series(recalibrate(frame, None, seed=seed), run_id, split, "platt"))
    elif calibration_frame is not None:
        series.append(
            _series(recalibrate(frame, calibration_frame, seed=seed), run_id, split, "platt")
        )
    else:
        logger.info(f"No validation inspection for {run_id}; {split} gets no Platt series.")
    return pd.concat(series, ignore_index=True)


def write_inspection_results(
    frame: pd.DataFrame,
    run_id: str,
    split: str,
    seed: int,
    extra: Optional[dict[str, Any]] = None,
    config_path: Optional[str] = None,
    runs_dir: Path = RUNS_DIR,
    write_inspection: bool = True,
) -> RunPaths:
    """Write the inspection CSV, its summary, the shared predictions and ``run.json``.

    ``write_inspection=False`` leaves an inspection CSV already in place
    untouched - the migration converts files it must not rewrite.
    """
    paths = RunPaths(run_id, runs_dir)
    paths.llm_dir.mkdir(parents=True, exist_ok=True)
    if write_inspection:
        frame.to_csv(paths.inspection(split), index=False)
    write_json(paths.llm_summary(split), summarize(frame))

    calibration_frame = None
    if split != "validation" and paths.inspection("validation").exists():
        calibration_frame = read_inspection_csv(paths.inspection("validation"))
    predictions = inspection_predictions(frame, run_id, split, calibration_frame, seed)
    write_predictions(predictions, run_id, split, runs_dir)

    try:
        info = read_run_info(run_id, runs_dir)
    except FileNotFoundError:
        info = RunInfo(
            run_id=run_id,
            family="llm",
            task="classification",
            selected_model=run_id,
            config_path=config_path,
            git_commit=git_commit(),
        )
    info.splits[split] = {"n_games": len(frame)}
    info.extra.update(extra or {})
    if (predictions["calibration"] == "platt").any():
        # The scoreboard reads an adapter after Platt; so does every comparison.
        info.selected_calibration = "platt"
        info.best_calibrator = "platt"
    write_run_info(info, runs_dir)
    return paths


def read_summary(run_id: str, split: str, runs_dir: Path = RUNS_DIR) -> dict[str, Any]:
    return json.loads(RunPaths(run_id, runs_dir).llm_summary(split).read_text())
