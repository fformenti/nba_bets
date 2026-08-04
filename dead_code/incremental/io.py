"""File discovery and loading for incremental ingestion."""

from pathlib import Path
from typing import Iterable

import pandas as pd

SUPPORTED_EXTENSIONS = (".csv", ".json", ".jsonl")


def discover_incremental_files(
    raw_dir: Path, max_files: int | None = None
) -> list[Path]:
    if not raw_dir.exists():
        return []

    files = [
        path
        for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    files.sort(key=lambda p: p.stat().st_mtime)
    if max_files is not None:
        return files[:max_files]
    return files


def read_incremental_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.read_json(path, orient="records")
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported file type: {path}")


def archive_files(files: Iterable[Path], archive_dir: Path) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        target = archive_dir / path.name
        path.replace(target)
