"""The result-file schema shared by every model family.

One row per game per (model, calibration) in ``predictions/<split>.csv``. The
sklearn experiment, the margin regressor and the LLM evaluation all write this
shape, which is what lets one chart put an xgboost classifier, a regressor and a
fine-tuned adapter on the same axes. Layout and field meanings:
docs/OUTPUTS.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1

SPLITS = ("validation", "test")

# "none" is the model's own output. The others name the calibrator applied on
# top: sigmoid/isotonic for sklearn, Platt for the LLM's first-token probability.
CALIBRATIONS = ("none", "sigmoid", "isotonic", "platt")

FAMILIES = ("sklearn", "llm")
TASKS = ("classification", "regression")

PREDICTION_COLUMNS = [
    "run_id",
    "model",
    "calibration",
    "split",
    "game_id",
    # 1 = home win. `winner` in the game tables is a teamId; this is not.
    "y_true",
    # The model's hard call, 1 = home. A regressor calls the sign of its margin.
    "y_pred",
    # P(home win). Empty for a model that emits no probability.
    "p_home",
    # Home minus away, in real points. Empty for a model that emits no margin.
    "pred_margin",
    "true_margin",
]

# The key a prediction row is unique on, within one split file.
SERIES_KEY = ["model", "calibration"]


def build_predictions(
    *,
    run_id: str,
    model: str,
    split: str,
    game_ids,
    y_true,
    y_pred,
    p_home=None,
    pred_margin=None,
    true_margin=None,
    calibration: str = "none",
) -> pd.DataFrame:
    """Assemble one (model, calibration) series into the shared schema."""
    n = len(game_ids)

    def column(values) -> np.ndarray:
        if values is None:
            return np.full(n, np.nan)
        return np.asarray(values, dtype=float).ravel()

    frame = pd.DataFrame(
        {
            "run_id": run_id,
            "model": model,
            "calibration": calibration,
            "split": split,
            "game_id": pd.to_numeric(pd.Series(game_ids).to_numpy()).astype("int64"),
            "y_true": np.asarray(y_true).astype(int).ravel(),
            "y_pred": np.asarray(y_pred).astype(int).ravel(),
            "p_home": column(p_home),
            "pred_margin": column(pred_margin),
            "true_margin": column(true_margin),
        }
    )
    return validate_predictions(frame)


def validate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Raise if ``frame`` breaks the schema; return it with columns in order."""
    missing = [c for c in PREDICTION_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Prediction frame is missing column(s): {missing}")

    bad_split = set(frame["split"]) - set(SPLITS)
    if bad_split:
        raise ValueError(f"Unknown split(s) {sorted(bad_split)}; expected one of {SPLITS}")
    bad_calibration = set(frame["calibration"]) - set(CALIBRATIONS)
    if bad_calibration:
        raise ValueError(
            f"Unknown calibration(s) {sorted(bad_calibration)}; expected one of {CALIBRATIONS}"
        )
    for column in ("y_true", "y_pred"):
        if not frame[column].isin([0, 1]).all():
            raise ValueError(f"{column} must be 0/1 (1 = home win)")
    p_home = frame["p_home"].dropna()
    if not p_home.between(0, 1).all():
        raise ValueError("p_home must lie in [0, 1]")
    if frame.duplicated(["split"] + SERIES_KEY + ["game_id"]).any():
        raise ValueError("A game appears twice for the same (split, model, calibration)")

    return frame[PREDICTION_COLUMNS]


@dataclass
class RunInfo:
    """``run.json``: what a run is, so a reader never has to infer it from a path."""

    run_id: str
    family: str
    task: str
    # The series the run stands for: the family that ships, or the adapter.
    selected_model: str
    selected_calibration: str = "none"
    # The calibrator validation preferred, whether or not it was adopted. The
    # calibration figure draws it against the uncalibrated model.
    best_calibrator: Optional[str] = None
    config_path: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None
    git_commit: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    # {split: {"n_games": int}}
    splits: dict[str, Any] = field(default_factory=dict)
    # Family-specific facts: hub_model_id, few_shot, n_features, ...
    extra: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}, got {self.family!r}")
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {self.task!r}")
        if self.selected_calibration not in CALIBRATIONS:
            raise ValueError(
                f"selected_calibration must be one of {CALIBRATIONS}, "
                f"got {self.selected_calibration!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunInfo":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
