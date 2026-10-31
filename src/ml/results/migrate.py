"""One-time move of ``outputs/`` into the run layout (docs/reference/outputs.md).

Nothing is deleted except macOS ``.DS_Store`` files, which would otherwise keep
emptied folders alive. Every output file is moved, and the LLM inspection CSVs
are additionally converted into the shared result schema - the only old outputs
that hold per-game predictions. The sklearn folders hold figures and aggregates
only, so they are archived as they are: there is nothing in them to convert.

Refuses to overwrite: if a destination already exists the migration stops
before moving anything.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.config.paths import OUTPUTS_DIR
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Old top-level folder -> archive folder, dated by the files' modification time.
ARCHIVED_FOLDERS = {
    "all_models": "2026-10-17_all_models",
    "xgboost": "2026-08-18_xgboost",
    "feature_audit": "2026-08-12_feature_audit",
    # configs/train/opus_search.yaml no longer exists.
    "opus_search": "2026-08-13_opus_search",
    # The season-decay backtest harness was deleted
    # (docs/findings/pipeline/2026-08-12-conference-split.md).
    "tuning": "2026-08-12_season_decay_tuning",
}

# The evaluation seed every LLM config uses; it seeds the Platt cross-fit.
LLM_EVALUATION_SEED = 42


class MigrationAborted(RuntimeError):
    pass


@dataclass
class Plan:
    moves: list[tuple[Path, Path]] = field(default_factory=list)
    llm_runs: list[str] = field(default_factory=list)

    def add(self, source: Path, destination: Path) -> None:
        self.moves.append((source, destination))


def plan_migration(outputs_dir: Path = OUTPUTS_DIR) -> Plan:
    """Every move the migration would make, without touching anything."""
    plan = Plan()
    archive = outputs_dir / "archive"
    for old, new in ARCHIVED_FOLDERS.items():
        if (outputs_dir / old).is_dir():
            plan.add(outputs_dir / old, archive / new)

    viz = outputs_dir / "enriched_games" / "viz"
    if viz.is_dir():
        for png in sorted(viz.glob("*.png")):
            plan.add(png, outputs_dir / "eda" / "home_win_ratio" / "figures" / png.name)

    eval_dir = outputs_dir / "llm" / "eval"
    if eval_dir.is_dir():
        for scoreboard in sorted(eval_dir.glob("scoreboard_*.csv")):
            plan.add(
                scoreboard,
                outputs_dir / "comparisons" / "llm_scoreboard" / "tables" / scoreboard.name,
            )
        for run_dir in sorted(p for p in eval_dir.iterdir() if p.is_dir()):
            run_root = outputs_dir / "runs" / run_dir.name
            plan.llm_runs.append(run_dir.name)
            for path in sorted(run_dir.iterdir()):
                if path.name == ".DS_Store":
                    continue
                if path.name.startswith("inspection_") and path.suffix == ".csv":
                    plan.add(path, run_root / "llm" / path.name)
                elif path.name == "mlflow.db":
                    plan.add(path, run_root / "tracking" / "mlflow.db")
                else:
                    # Anything else an eval folder holds (old HTML charts, say)
                    # is kept beside the run's other LLM-only files.
                    plan.add(path, run_root / "llm" / path.name)

    # Checkpoints, if a training box ever left any here.
    qlora = outputs_dir / "llm" / "qlora"
    if qlora.is_dir():
        for model_dir in sorted(p for p in qlora.iterdir() if p.is_dir()):
            for run_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                plan.add(run_dir, outputs_dir / "checkpoints" / run_dir.name)
    return plan


def _remove_empty_dirs(root: Path, outputs_dir: Path) -> None:
    """Remove ``root`` and its now-empty subfolders, ignoring .DS_Store."""
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.name == ".DS_Store":
            path.unlink()
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    if root.is_dir() and not any(p for p in root.iterdir() if p.name != ".DS_Store"):
        shutil.rmtree(root)
    logger.debug(f"Cleaned {root.relative_to(outputs_dir)}")


def migrate_outputs_layout(outputs_dir: Path = OUTPUTS_DIR, dry_run: bool = False) -> Plan:
    """Move ``outputs_dir`` into the run layout and convert the LLM evaluations."""
    from src.ml.llm.diagnostics import read_inspection_csv
    from src.ml.llm.results import write_inspection_results
    from src.ml.results.schema import RunInfo
    from src.ml.results.store import write_run_info

    plan = plan_migration(outputs_dir)
    clashes = [str(dst) for _, dst in plan.moves if dst.exists()]
    if clashes:
        raise MigrationAborted(f"Destination(s) already exist: {clashes}")

    for src, dst in plan.moves:
        logger.info(f"{'[dry-run] ' if dry_run else ''}{src.relative_to(outputs_dir)} -> {dst.relative_to(outputs_dir)}")
    if dry_run:
        logger.info(f"[dry-run] would convert {len(plan.llm_runs)} LLM run(s): {plan.llm_runs}")
        return plan

    for src, dst in plan.moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

    runs_dir = outputs_dir / "runs"
    for run_id in plan.llm_runs:
        llm_dir = runs_dir / run_id / "llm"
        csvs = {p.stem.removeprefix("inspection_"): p for p in llm_dir.glob("inspection_*.csv")}
        if not csvs:
            logger.info(f"{run_id}: no inspection CSV; only moved.")
            continue
        # Dated by the evaluation itself, not by today, so `latest` still
        # means the latest run actually made.
        created = datetime.fromtimestamp(min(p.stat().st_mtime for p in csvs.values()))
        write_run_info(
            RunInfo(
                run_id=run_id,
                family="llm",
                task="classification",
                selected_model=run_id,
                created_at=created.isoformat(timespec="seconds"),
                extra={"migrated_from": f"outputs/llm/eval/{run_id}"},
            ),
            runs_dir,
        )
        # Validation first: test's Platt fit is read from it.
        for split in sorted(csvs, key=lambda s: s != "validation"):
            write_inspection_results(
                read_inspection_csv(csvs[split]),
                run_id=run_id,
                split=split,
                seed=LLM_EVALUATION_SEED,
                runs_dir=runs_dir,
                write_inspection=False,
            )
        logger.info(f"{run_id}: converted {sorted(csvs)}")

    for leftover in ("llm", "enriched_games"):
        _remove_empty_dirs(outputs_dir / leftover, outputs_dir)
    return plan
