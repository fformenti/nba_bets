"""Draw a run's figures, or several runs' on shared axes, from result files alone.

Nothing here trains or loads a model. Inputs are a run's ``predictions/``,
``run.json`` and ``model/`` files, the game table for metadata (team, season,
conference, games played, score) and the Polymarket odds table. Everything
written lands in ``outputs/runs/<run_id>/report/`` or
``outputs/comparisons/<name>/`` and can be deleted and redrawn at will.
Layout and schema: docs/reference/outputs.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config.paths import (
    COMPARISONS_DIR,
    REGULAR_SEASON_GAMES_FEATURES_PATH,
    RUNS_DIR,
    RunPaths,
)
from src.ml.evaluation.polymarket_baseline import (
    align_market_to_rows,
    build_scored_frame,
    compare_to_market,
)
from src.ml.models.baseline import RecordDifferenceBaseline
from src.ml.plotting import breakdowns, charts
from src.ml.plotting.llm import series_figures
from src.ml.results import (
    RunInfo,
    read_predictions,
    read_run_info,
    score_predictions,
    score_series,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# The game-level facts the breakdown figures group by. Read from the game
# table by gameId at plot time: they describe games, not runs, so an LLM run
# gets the same breakdowns as an sklearn run without carrying them itself.
METADATA_COLUMNS = [
    "gameId",
    "season",
    "hometeamName",
    "awayteamName",
    "hometeamConference",
    "awayteamConference",
    "overtimes",
    "is_neutral_court_game",
    "homeScore",
    "awayScore",
    "games_played_HT",
    "games_played_VT",
    # Season-to-date win% (L82 never reaches past the current season's games),
    # read by the record baseline in build_comparison.
    "record_L82_HT",
    "record_L82_VT",
]


def load_game_metadata(
    game_ids, features_path: Path = REGULAR_SEASON_GAMES_FEATURES_PATH
) -> pd.DataFrame:
    """Metadata rows for ``game_ids``, in that order (missing games are NaN rows)."""
    available = pd.read_csv(features_path, nrows=0).columns
    table = pd.read_csv(features_path, usecols=[c for c in METADATA_COLUMNS if c in available])
    table = table.drop_duplicates(subset="gameId").set_index("gameId")
    ids = pd.Index(pd.to_numeric(pd.Series(game_ids)).astype("int64"), name="gameId")
    return table.reindex(ids).reset_index()


def _metadata_or_none(game_ids) -> Optional[pd.DataFrame]:
    """Game metadata, or ``None`` (breakdowns skipped) on a box without the game table."""
    try:
        return load_game_metadata(game_ids)
    except FileNotFoundError as exc:
        logger.warning(f"No game table, so no breakdown figures: {exc}")
        return None


@dataclass
class Series:
    """One (run, model, calibration) slice of predictions, in game order."""

    label: str
    frame: pd.DataFrame

    @property
    def y_true(self) -> pd.Series:
        return self.frame["y_true"].reset_index(drop=True)

    @property
    def y_pred(self) -> np.ndarray:
        return self.frame["y_pred"].to_numpy()

    @property
    def p_home(self) -> Optional[np.ndarray]:
        p = self.frame["p_home"]
        return p.to_numpy() if len(p) and p.notna().all() else None

    def on(self, game_ids) -> "Series":
        """This series restricted to, and ordered by, ``game_ids``."""
        frame = self.frame.set_index("game_id").loc[list(game_ids)].reset_index()
        return Series(self.label, frame[self.frame.columns])


def _select(
    predictions: pd.DataFrame, model: str, calibration: str, split: str
) -> pd.DataFrame:
    return predictions[
        (predictions["model"] == model)
        & (predictions["calibration"] == calibration)
        & (predictions["split"] == split)
    ].reset_index(drop=True)


class _Writer:
    """Saves figures and tables into one report folder and remembers what it wrote."""

    def __init__(self, figures_dir: Path, tables_dir: Path):
        self.figures_dir = figures_dir
        self.tables_dir = tables_dir
        self.written: list[Path] = []

    def figure(self, name: str, fig) -> None:
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(fig, "write_html"):
            path = self.figures_dir / f"{name}.html"
            fig.write_html(path)
        else:
            path = self.figures_dir / f"{name}.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        self.written.append(path)

    def table(self, name: str, table: pd.DataFrame) -> None:
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        path = self.tables_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        self.written.append(path)

    def attempt(self, name: str, draw: Callable[[], None]) -> None:
        """Run one figure's drawing; a figure that cannot be drawn is skipped, not fatal."""
        try:
            draw()
        except Exception as exc:
            logger.warning(f"Skipped {name}: {exc}")


def default_split(run_id: str, runs_dir: Path = RUNS_DIR) -> str:
    """Test when the run scored it, else validation (an LLM run usually has only that)."""
    return "test" if RunPaths(run_id, runs_dir).predictions("test").exists() else "validation"


def build_run_report(
    run_id: str, split: Optional[str] = None, runs_dir: Path = RUNS_DIR
) -> list[Path]:
    """Draw every figure the run's files support into ``report/``. Returns what was written."""
    info = read_run_info(run_id, runs_dir)
    split = split or default_split(run_id, runs_dir)
    predictions = read_predictions(run_id, [split], runs_dir)
    paths = RunPaths(run_id, runs_dir)
    out = _Writer(paths.figures_dir, paths.tables_dir)

    model = info.selected_model
    shipped = _select(predictions, model, info.selected_calibration, split)
    if shipped.empty:
        shipped = _select(predictions, model, "none", split)
    if shipped.empty:
        raise ValueError(f"Run {run_id!r} has no {split} predictions for {model!r}")
    raw = _select(predictions, model, "none", split)
    calibrated = (
        _select(predictions, model, info.best_calibrator, split)
        if info.best_calibrator and info.best_calibrator != "none"
        else pd.DataFrame()
    )

    selected = Series(model, shipped)
    metadata = _metadata_or_none(shipped["game_id"])
    suffix = f" - {model} ({split}, n={len(shipped)})"
    logger.info(f"Drawing {run_id} / {model} [{info.selected_calibration}] on {split}")

    out.table("metrics", score_predictions(predictions))
    _classification_figures(out, selected, raw, calibrated, info, suffix)
    if metadata is not None:
        _breakdown_figures(out, selected, predictions, split, metadata, suffix)
    _model_figures(out, paths, model)
    _market_figures(out, selected, raw, calibrated, info, predictions, split, metadata, model)
    if info.task == "regression":
        _regression_figures(out, selected, suffix)
    if info.family == "llm":
        # The raw model: Platt moves only the probability, and the margin
        # figures need the margins only the raw series carries.
        for name, fig in series_figures(raw if not raw.empty else shipped, f"{run_id} ({split})").items():
            out.figure(name, fig)

    logger.info(f"Wrote {len(out.written)} file(s) under {paths.report_dir}")
    return out.written


def _classification_figures(out, selected, raw, calibrated, info, suffix) -> None:
    out.attempt(
        "confusion_matrix",
        lambda: out.figure(
            "confusion_matrix",
            charts.plot_confusion_matrix(
                selected.y_true, selected.y_pred, class_names=[0, 1],
                title=f"Confusion Matrix{suffix}",
            ),
        ),
    )
    if selected.p_home is None:
        return
    out.figure(
        "roc_curve", charts.plot_roc_curve(selected.y_true, selected.p_home, title=f"ROC Curve{suffix}")
    )
    out.figure(
        "prediction_accuracy_by_bin",
        charts.plot_prediction_accuracy_by_bin(
            selected.y_true, selected.p_home,
            title=f"Prediction Accuracy by Probability Bin{suffix}",
        ),
    )
    if not raw.empty and raw["p_home"].notna().all():
        raw_series = Series("raw", raw).on(selected.frame["game_id"])
        cal_proba = (
            Series("cal", calibrated).on(selected.frame["game_id"]).p_home
            if not calibrated.empty
            else None
        )
        out.figure(
            "calibration_curve",
            charts.plot_calibration_curve(
                selected.y_true, raw_series.p_home, cal_proba,
                calibration_method=info.best_calibrator or "sigmoid",
                title=f"Calibration Curve{suffix}",
            ),
        )


def _breakdown_figures(out, selected, predictions, split, metadata, suffix) -> None:
    y_true, y_pred = selected.y_true, selected.y_pred
    record = _select(predictions, "record_baseline", "none", split)
    comparisons = (
        {"Record baseline": Series("rb", record).on(selected.frame["game_id"]).y_pred}
        if not record.empty
        else None
    )

    figures = {
        "error_by_team": lambda: breakdowns.plot_error_by_team(
            y_true, y_pred, metadata, title=f"Model Error Rate by Team{suffix}"
        ),
        "accuracy_by_season": lambda: breakdowns.plot_accuracy_by_season(
            y_true, y_pred, metadata, title=f"Model Accuracy by Season{suffix}"
        ),
        "conference_matchup_accuracy": lambda: breakdowns.plot_conference_matchup_accuracy(
            y_true, y_pred, metadata, title=f"Accuracy by Conference Matchup{suffix}"
        ),
        "overtime_accuracy": lambda: breakdowns.plot_overtime_accuracy(
            y_true, y_pred, metadata, title=f"Accuracy: Regular vs Overtime{suffix}"
        ),
        "neutral_court_accuracy": lambda: breakdowns.plot_neutral_court_accuracy(
            y_true, y_pred, metadata, title=f"Accuracy: Home vs Neutral Court{suffix}"
        ),
        "season_phase_accuracy": lambda: breakdowns.plot_season_phase_accuracy(
            y_true, y_pred, metadata, comparison_preds=comparisons,
            title=f"Accuracy by Season Phase{suffix}",
        ),
    }
    if selected.p_home is not None:
        figures["score_margin_calibration"] = lambda: breakdowns.plot_score_margin_calibration(
            y_true, selected.p_home, metadata,
            title=f"Predicted Probability vs Score Margin{suffix}",
        )

    for name, draw in figures.items():
        def save(name=name, draw=draw):
            fig, table = draw()
            out.figure(name, fig)
            out.table(name, table)

        out.attempt(name, save)


def _model_figures(out, paths: RunPaths, model: str) -> None:
    if paths.feature_selection.exists():
        results = json.loads(paths.feature_selection.read_text())
        out.figure(
            "boruta_shap_selection",
            charts.plot_boruta_shap_results(results, top_n=50, title="Boruta-SHAP Feature Selection"),
        )
    if paths.feature_importance.exists():
        out.figure(
            "feature_importance",
            charts.plot_feature_importance(
                pd.read_csv(paths.feature_importance), top_n=20,
                title=f"Feature Importance - {model}",
            ),
        )
    if paths.shap_values.exists():
        out.attempt(
            "shap_summary",
            lambda: out.figure(
                "shap_summary",
                charts.plot_shap_summary(
                    pd.read_csv(paths.shap_values),
                    title=f"SHAP Feature Contribution - {model}", top_n=20,
                ),
            ),
        )


def _market_figures(out, selected, raw, calibrated, info, predictions, split, metadata, model) -> None:
    """Model-vs-Polymarket twins, drawn only on the games the market priced.

    Separate files rather than extra lines on the figures above: the odds
    backfill starts at 2024/25 and drops markets that never traded, so the
    market prices only part of the split, and the game count goes in each
    title so the two can never be read as one.
    """
    try:
        covered, market_proba = align_market_to_rows(selected.frame["game_id"])
    except FileNotFoundError as exc:
        logger.info(f"Skipping Polymarket figures: {exc}")
        return
    n_covered = int(covered.sum())
    if n_covered == 0:
        logger.info(f"No {split} game is priced by Polymarket; skipping market figures.")
        return

    ids = selected.frame["game_id"][covered]
    shared = selected.on(ids)
    suffix = f" - {model} (n={n_covered} shared games)"

    out.attempt(
        "polymarket_baseline",
        lambda: out.table(
            "polymarket_baseline",
            compare_to_market(
                build_scored_frame(ids, shared.y_true, shared.y_pred, shared.p_home)
            ),
        ),
    )
    if shared.p_home is not None:
        out.figure(
            "roc_curve_vs_market",
            charts.plot_roc_curve(
                shared.y_true, shared.p_home,
                title=f"ROC Curve vs Polymarket{suffix}", model_label=model,
                reference_curves={"Polymarket": market_proba},
            ),
        )
    if not raw.empty and raw["p_home"].notna().all():
        out.figure(
            "calibration_curve_vs_market",
            charts.plot_calibration_curve(
                shared.y_true,
                Series("raw", raw).on(ids).p_home,
                Series("cal", calibrated).on(ids).p_home if not calibrated.empty else None,
                calibration_method=info.best_calibrator or "sigmoid",
                y_pred_proba_market=market_proba,
                title=f"Calibration Curve vs Polymarket{suffix}",
                # Three near-coincident curves; the bin counts stay on the
                # two-series `calibration_curve.png` next to this one.
                annotate_counts=False,
            ),
        )

    # The market's hard call is just its favourite, as in `compare_to_market`.
    comparisons = {"Polymarket": (market_proba >= 0.5).astype(int)}
    record = _select(predictions, "record_baseline", "none", split)
    if not record.empty:
        comparisons["Record baseline"] = Series("rb", record).on(ids).y_pred

    def season_phase():
        if metadata is None:
            return
        fig, table = breakdowns.plot_season_phase_accuracy(
            shared.y_true, shared.y_pred, metadata[covered].reset_index(drop=True),
            comparison_preds=comparisons,
            title=f"Accuracy by Season Phase vs Polymarket{suffix}",
        )
        out.figure("season_phase_accuracy_vs_market", fig)
        out.table("season_phase_accuracy_vs_market", table)

    out.attempt("season_phase_accuracy_vs_market", season_phase)


def _regression_figures(out, selected, suffix) -> None:
    margins = selected.frame[["true_margin", "pred_margin"]].dropna()
    if margins.empty:
        return
    out.figure(
        "predictions_vs_actual",
        charts.plot_predictions(
            margins["true_margin"], margins["pred_margin"].to_numpy(),
            title=f"Predicted vs Actual Margin{suffix}",
        ),
    )
    out.figure(
        "residuals",
        charts.plot_residuals(
            margins["true_margin"], margins["pred_margin"].to_numpy(),
            title=f"Margin Residuals{suffix}",
        ),
    )


# ---------------------------------------------------------------------------
# Several runs on shared axes
# ---------------------------------------------------------------------------


def parse_series_spec(spec: str, runs_dir: Path = RUNS_DIR) -> tuple[RunInfo, str, str]:
    """``run_id[:model[:calibration]]``; omitted parts default to what run.json selected."""
    run_id, *rest = spec.split(":")
    info = read_run_info(run_id, runs_dir)
    model = rest[0] if len(rest) > 0 and rest[0] else info.selected_model
    calibration = rest[1] if len(rest) > 1 else (
        info.selected_calibration if model == info.selected_model else "none"
    )
    return info, model, calibration


def record_baseline_calls(metadata: pd.DataFrame) -> np.ndarray:
    """Pick the team with the better season-to-date record; a tie goes home.

    The same rule as the sklearn runs' ``record_baseline`` series, but computed
    from the game table for any gameIds, so it covers games no sklearn run
    scored (an LLM built with a different games-played filter, say).
    """
    delta = metadata["record_L82_HT"] - metadata["record_L82_VT"]
    if delta.isna().any():
        raise ValueError(f"{int(delta.isna().sum())} game(s) have no season record in the game table")
    X = pd.DataFrame({"record_L82_delta": delta.to_numpy()})
    return RecordDifferenceBaseline().fit(X).predict(X)


def build_comparison(
    specs: list[str],
    name: str,
    split: str = "validation",
    include_market: bool = False,
    include_record: bool = False,
    runs_dir: Path = RUNS_DIR,
    comparisons_dir: Path = COMPARISONS_DIR,
) -> list[Path]:
    """Put several series - any family - on the same games and the same axes.

    Every series is restricted to the gameIds all of them scored, so a
    difference on the figures comes from the predictions and never from two
    slates of games. ``include_market`` adds Polymarket as one more series,
    which narrows the slate to the games it priced (test seasons only).
    ``include_record`` adds the record baseline, a hard call with no
    probability, so it scores accuracy only.
    """
    series: list[Series] = []
    # (run, model) pairs that also appear calibrated: their raw curve is drawn
    # dashed, as the twin of the calibrated one.
    keys: list[tuple[str, str, str]] = []
    for spec in specs:
        info, model, calibration = parse_series_spec(spec, runs_dir)
        frame = _select(read_predictions(info.run_id, [split], runs_dir), model, calibration, split)
        if frame.empty:
            raise ValueError(f"{spec!r}: no {split} rows for model {model!r} [{calibration}]")
        label = model if model == info.run_id else f"{info.run_id}:{model}"
        if calibration != "none":
            label += f" [{calibration}]"
        series.append(Series(label, frame))
        keys.append((info.run_id, model, calibration))

    shared = set(series[0].frame["game_id"])
    for s in series[1:]:
        shared &= set(s.frame["game_id"])
    ids = sorted(shared)

    market_proba = None
    if include_market:
        covered, prices = align_market_to_rows(pd.Series(ids))
        if covered.any():
            ids = list(np.asarray(ids)[covered])
            market_proba = prices
        else:
            logger.warning(f"Polymarket priced none of the shared {split} games; leaving it out.")
    if not ids:
        raise ValueError("The series share no games; nothing to compare.")

    aligned = [s.on(ids) for s in series]
    y_true = aligned[0].y_true
    for s in aligned[1:]:
        if not (s.y_true.to_numpy() == y_true.to_numpy()).all():
            raise ValueError(f"{s.label} disagrees with {aligned[0].label} on outcomes")

    metadata = _metadata_or_none(ids)
    record_calls = None
    if include_record:
        if metadata is None:
            raise ValueError("The record baseline needs the game table, and there is none here.")
        record_calls = record_baseline_calls(metadata)

    folder = comparisons_dir / name
    out = _Writer(folder / "figures", folder / "tables")
    suffix = f" ({split}, n={len(ids)} shared games)"

    rows = []
    for spec, original, s in zip(specs, series, aligned):
        rows.append(
            {"series": s.label, "spec": spec, "n_games_own": len(original.frame), **score_series(s.frame)}
        )
    if market_proba is not None:
        market = pd.DataFrame(
            {
                "y_true": y_true, "y_pred": (market_proba >= 0.5).astype(int),
                "p_home": market_proba, "pred_margin": np.nan, "true_margin": np.nan,
            }
        )
        rows.append({"series": "Polymarket", "spec": "market", "n_games_own": len(ids), **score_series(market)})
    if record_calls is not None:
        record = pd.DataFrame(
            {
                "y_true": y_true, "y_pred": record_calls,
                "p_home": np.nan, "pred_margin": np.nan, "true_margin": np.nan,
            }
        )
        rows.append({"series": "Record baseline", "spec": "record", "n_games_own": len(ids), **score_series(record)})
    out.table("scoreboard", pd.DataFrame(rows))

    probabilistic = {s.label: s.p_home for s in aligned if s.p_home is not None}
    if market_proba is not None:
        probabilistic["Polymarket"] = market_proba
    if probabilistic:
        first, *others = probabilistic.items()
        out.figure(
            "roc_curve",
            charts.plot_roc_curve(
                y_true, first[1], title=f"ROC Curve{suffix}", model_label=first[0],
                reference_curves=dict(others),
            ),
        )
        out.figure(
            "calibration_curve",
            charts.plot_reliability_comparison(
                y_true, probabilistic, title=f"Calibration{suffix}",
                dashed={
                    s.label
                    for s, (run, model, calibration) in zip(aligned, keys)
                    if calibration == "none"
                    and any(k[:2] == (run, model) and k[2] != "none" for k in keys)
                },
            ),
        )

    hard_calls = {s.label: s.y_pred for s in aligned}
    if market_proba is not None:
        hard_calls["Polymarket"] = (market_proba >= 0.5).astype(int)
    if record_calls is not None:
        hard_calls["Record baseline"] = record_calls
    (first_label, first_pred), *other_calls = hard_calls.items()

    def season_phase():
        fig, table = breakdowns.plot_season_phase_accuracy(
            y_true, first_pred, metadata, comparison_preds=dict(other_calls),
            title=f"Accuracy by Season Phase: {first_label} vs others{suffix}",
        )
        out.figure("season_phase_accuracy", fig)
        out.table("season_phase_accuracy", table)

    if metadata is not None:
        out.attempt("season_phase_accuracy", season_phase)
    logger.info(f"Wrote {len(out.written)} file(s) under {folder}")
    return out.written


def log_report_to_mlflow(run_id: str, runs_dir: Path = RUNS_DIR) -> bool:
    """Attach ``report/`` to the run's MLflow run, if it has one. Returns whether it did."""
    info = read_run_info(run_id, runs_dir)
    report_dir = RunPaths(run_id, runs_dir).report_dir
    if not info.mlflow_run_id or not report_dir.exists():
        return False
    from mlflow.tracking import MlflowClient

    MlflowClient(tracking_uri=info.mlflow_tracking_uri).log_artifacts(
        info.mlflow_run_id, str(report_dir), artifact_path="report"
    )
    logger.info(f"Logged {report_dir} to MLflow run {info.mlflow_run_id}")
    return True
