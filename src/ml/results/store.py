"""Read and write the files under ``outputs/runs/<run_id>/``.

Training and evaluation write through here; the plotting module reads through
here. Neither builds a path itself: they come from
:class:`src.config.paths.RunPaths`.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from src.config.paths import PROJECT_ROOT, RUNS_DIR, RunPaths
from src.ml.results.schema import (
    SCHEMA_VERSION,
    SERIES_KEY,
    SPLITS,
    RunInfo,
    validate_predictions,
)
from src.ml.results.scoring import score_predictions
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def git_commit() -> Optional[str]:
    """Short SHA of HEAD, or ``None`` outside a git checkout (a Colab copy, say)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _json_safe(value: Any) -> Any:
    """NaN is not JSON; write it as null."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, default=str) + "\n")


def write_run_info(info: RunInfo, runs_dir: Path = RUNS_DIR) -> Path:
    path = RunPaths(info.run_id, runs_dir).run_json
    write_json(path, info.to_dict())
    return path


def read_run_info(run_id: str, runs_dir: Path = RUNS_DIR) -> RunInfo:
    path = RunPaths(run_id, runs_dir).run_json
    if not path.exists():
        raise FileNotFoundError(f"No run.json for run {run_id!r} at {path}")
    return RunInfo.from_dict(json.loads(path.read_text()))


def write_predictions(
    frame: pd.DataFrame, run_id: str, split: str, runs_dir: Path = RUNS_DIR
) -> Path:
    """Upsert ``frame`` into ``predictions/<split>.csv`` and rescore the split.

    Rows are replaced per (model, calibration), not per file: the LLM writes
    validation and test in separate invocations, and re-scoring one adapter
    must not drop a series another command wrote to the same split.
    """
    frame = validate_predictions(frame)
    if set(frame["split"]) != {split} or set(frame["run_id"]) != {run_id}:
        raise ValueError(f"Frame rows must all belong to run {run_id!r}, split {split!r}")

    paths = RunPaths(run_id, runs_dir)
    path = paths.predictions(split)
    if path.exists():
        existing = pd.read_csv(path)
        incoming = frame[SERIES_KEY].drop_duplicates()
        keep = existing.merge(incoming, on=SERIES_KEY, how="left", indicator=True)
        existing = existing[(keep["_merge"] == "left_only").to_numpy()]
        frame = pd.concat([existing, frame], ignore_index=True)
    frame = validate_predictions(frame)

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    write_metrics(frame, run_id, split, runs_dir)
    logger.info(f"Predictions written to {path} ({len(frame)} rows)")
    return path


def write_metrics(frame: pd.DataFrame, run_id: str, split: str, runs_dir: Path = RUNS_DIR) -> Path:
    path = RunPaths(run_id, runs_dir).metrics(split)
    write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "split": split,
            "rows": score_predictions(frame).to_dict(orient="records"),
        },
    )
    return path


def read_predictions(
    run_id: str, splits: Optional[Iterable[str]] = None, runs_dir: Path = RUNS_DIR
) -> pd.DataFrame:
    """Every prediction row of ``run_id`` for ``splits`` (default: whichever exist)."""
    paths = RunPaths(run_id, runs_dir)
    frames = [
        pd.read_csv(paths.predictions(split))
        for split in (splits or SPLITS)
        if paths.predictions(split).exists()
    ]
    if not frames:
        raise FileNotFoundError(f"Run {run_id!r} has no predictions under {paths.root}")
    return validate_predictions(pd.concat(frames, ignore_index=True))


def read_metrics(run_id: str, split: str, runs_dir: Path = RUNS_DIR) -> pd.DataFrame:
    path = RunPaths(run_id, runs_dir).metrics(split)
    return pd.DataFrame(json.loads(path.read_text())["rows"])


def list_runs(runs_dir: Path = RUNS_DIR) -> list[RunInfo]:
    """Every run with a ``run.json``, oldest first."""
    infos = [
        RunInfo.from_dict(json.loads(path.read_text()))
        for path in sorted(runs_dir.glob("*/run.json"))
    ]
    return sorted(infos, key=lambda info: info.created_at)


def latest_run_id(runs_dir: Path = RUNS_DIR, family: Optional[str] = None) -> str:
    runs = [info for info in list_runs(runs_dir) if family in (None, info.family)]
    if not runs:
        raise FileNotFoundError(f"No runs under {runs_dir}")
    return runs[-1].run_id
