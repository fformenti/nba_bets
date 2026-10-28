"""Keep selected repo paths in step with ``My Drive/nba_bets/`` for Colab.

Two directions, because the paths have two owners:

- **Code and config** (``PUSH_PATHS``) are written only here. Push mirrors them:
  new and changed files go up, and files that exist only on Drive are moved to
  Drive's trash. The trash is not optional — Colab ``rsync --delete``s the Drive
  copy onto its local disk, so a module deleted here but left on Drive would
  come back to life there.
- **Results** (``PULL_PATHS``) land on Drive from Colab and are forwarded
  here. Pull only downloads: nothing is uploaded or deleted on either side, and
  figures and images are skipped (they are redrawn locally from the tables).
  When both sides changed a file, a newer Drive copy overwrites the local one; a
  newer local copy is left alone. Downloads carry the mtime across, which is
  what keeps that comparison meaningful on the next run.

Change detection is md5: Drive reports ``md5Checksum`` for every binary file,
so an unchanged file costs one listing entry and no transfer.

Planning (:func:`plan_push`, :func:`plan_pull`) is pure and is what the
tests exercise; :func:`apply_actions` is the only part that writes. The planner
only ever receives paths under the scopes, so nothing else in ``nba_bets/`` on
Drive can be touched.
"""

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Literal

from src.gdrive import drive
from src.gdrive.drive import DriveFile

logger = logging.getLogger(__name__)

DRIVE_ROOT_NAME = "nba_bets"
PUSH_PATHS = ("src", "configs", "colab_notebooks", "pyproject.toml")
# Listed one by one rather than as outputs/: outputs/checkpoints/ can hold
# multi-GB adapters, and a new outputs/ subfolder should be a decision.
# outputs/eda and outputs/archive are produced locally only, so they are not pulled.
PULL_PATHS = ("outputs/runs", "outputs/comparisons")
SCOPE_GROUPS = {"code": PUSH_PATHS, "outputs": PULL_PATHS}

IGNORED_NAMES = frozenset({".DS_Store", "__pycache__", ".ipynb_checkpoints", "Icon\r"})
IGNORED_SUFFIXES = frozenset({".pyc"})
# Pull skips plots: they live in */figures/ (png and plotly html) and are redrawn locally.
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp"})
# Drive keeps ms, the local filesystem keeps ns; closer than this is "same time".
TIE_TOLERANCE_SECONDS = 2.0

Kind = Literal[
    "upload",
    "update",
    "trash",
    "download",
    "overwrite_local",
    "skip",
    "unchanged",
]
# Display order and wording for format_plan.
KIND_LABELS: dict[str, str] = {
    "upload": "UPLOAD          (new on Drive)",
    "update": "UPDATE          (Drive copy overwritten)",
    "download": "DOWNLOAD        (new locally)",
    "overwrite_local": "OVERWRITE LOCAL (Drive is newer)",
    "trash": "TRASH ON DRIVE  (recoverable 30 days)",
    "skip": "SKIP            (needs a human)",
}


@dataclass(frozen=True)
class LocalFile:
    md5: str
    size: int
    mtime: datetime


@dataclass(frozen=True)
class Action:
    kind: Kind
    rel_path: str
    reason: str
    remote: DriveFile | None = None


def is_ignored(rel_path: str) -> bool:
    parts = PurePosixPath(rel_path).parts
    return any(part in IGNORED_NAMES for part in parts) or PurePosixPath(rel_path).suffix in (
        IGNORED_SUFFIXES
    )


def is_figure(rel_path: str) -> bool:
    path = PurePosixPath(rel_path)
    return any(part.endswith("figures") for part in path.parent.parts) or (
        path.suffix.lower() in IMAGE_SUFFIXES
    )


# ─── Indexing ────────────────────────────────────────────────────────────────


def local_index(repo_root: Path, scope: str) -> dict[str, LocalFile]:
    """Every non-junk file under a scope, keyed by repo-relative posix path."""
    target = repo_root / scope
    paths = [target] if target.is_file() else sorted(p for p in target.rglob("*") if p.is_file())
    index: dict[str, LocalFile] = {}
    for path in paths:
        rel = path.relative_to(repo_root).as_posix()
        if is_ignored(rel):
            continue
        stat = path.stat()
        index[rel] = LocalFile(
            md5=drive.md5_of(path),
            size=stat.st_size,
            mtime=datetime.fromtimestamp(stat.st_mtime).astimezone(),
        )
    return index


def remote_index(service, drive_root_id: str, scope: str) -> dict[str, list[DriveFile]]:
    """Every file under a scope on Drive, keyed by the same repo-relative paths."""
    parent, _, name = scope.rpartition("/")
    parent_id = drive_root_id
    if parent:
        node = drive.resolve_path(service, parent, drive_root_id)
        if node is None:
            return {}
        parent_id = node.id
    # Listing the parent rather than find_child keeps duplicate names visible.
    matches = [f for f in drive.list_folder(service, parent_id) if f.name == name]
    index: dict[str, list[DriveFile]] = {}
    for match in matches:
        if match.is_folder:
            for rel, items in drive.walk_folder(service, match.id, f"{scope}/").items():
                index.setdefault(rel, []).extend(items)
        else:
            index.setdefault(scope, []).append(match)
    return index


# ─── Planning (pure) ─────────────────────────────────────────────────────────


def _dirs(paths) -> set[str]:
    return {str(parent) for p in paths for parent in PurePosixPath(p).parents if str(parent) != "."}


def _unusable(rel: str, items: list[DriveFile]) -> Action | None:
    """A Drive entry the sync must not guess about, as a skip; else None."""
    if len(items) > 1:
        return Action("skip", rel, f"{len(items)} files with this name on Drive")
    if items[0].is_google_native:
        return Action("skip", rel, f"Google-native file ({items[0].mime_type})")
    if items[0].md5 is None:
        return Action("skip", rel, "Drive reports no md5")
    return None


def _shape_clashes(
    local: dict[str, LocalFile], remote: dict[str, list[DriveFile]]
) -> tuple[list[Action], set[str]]:
    """Paths that are a file on one side and a folder on the other.

    Both sides of a clash are skipped — the file and everything under the
    folder — so a push never trashes a folder's contents to make room.
    """

    def under_file(path: str, files) -> bool:
        return any(str(parent) in files for parent in PurePosixPath(path).parents)

    local_dirs, remote_dirs = _dirs(local), _dirs(remote)
    clashes = [
        Action("skip", p, "file locally, folder on Drive") for p in local if p in remote_dirs
    ]
    clashes += [
        Action("skip", p, "folder locally, file on Drive") for p in remote if p in local_dirs
    ]
    clashes += [
        Action("skip", p, "under a local file's name") for p in remote if under_file(p, local)
    ]
    clashes += [
        Action("skip", p, "under a Drive file's name") for p in local if under_file(p, remote)
    ]
    return clashes, {a.rel_path for a in clashes}


def plan_push(local: dict[str, LocalFile], remote: dict[str, list[DriveFile]]) -> list[Action]:
    """Make Drive equal the repo. Local is the truth; Drive-only files go to trash."""
    actions, clashed = _shape_clashes(local, remote)
    for rel in sorted(set(local) | set(remote)):
        if rel in clashed:
            continue
        items = remote.get(rel)
        if rel not in local:
            if is_ignored(rel):
                actions.extend(Action("trash", rel, "macOS/Python junk", item) for item in items)
            elif all(not item.is_google_native for item in items):
                actions.extend(Action("trash", rel, "not in the repo", item) for item in items)
            else:
                actions.append(Action("skip", rel, "Google-native file not in the repo"))
            continue
        if items is None:
            actions.append(Action("upload", rel, "only in the repo"))
            continue
        problem = _unusable(rel, items)
        if problem is not None:
            actions.append(problem)
        elif items[0].md5 == local[rel].md5:
            actions.append(Action("unchanged", rel, "same md5", items[0]))
        else:
            actions.append(Action("update", rel, "content differs", items[0]))
    return actions


def plan_pull(local: dict[str, LocalFile], remote: dict[str, list[DriveFile]]) -> list[Action]:
    """Bring Drive's files down. Never uploads or deletes; a newer local copy is kept."""
    remote = {
        rel: items for rel, items in remote.items() if not (is_ignored(rel) or is_figure(rel))
    }
    actions, clashed = _shape_clashes(local, remote)
    for rel in sorted(remote):
        if rel in clashed:
            continue
        items = remote[rel]
        problem = _unusable(rel, items)
        if problem is not None:
            actions.append(problem)
            continue
        item = items[0]
        if rel not in local:
            actions.append(Action("download", rel, "only on Drive", item))
            continue
        if item.md5 == local[rel].md5:
            actions.append(Action("unchanged", rel, "same md5", item))
            continue
        delta = (item.modified_time - local[rel].mtime).total_seconds()
        if abs(delta) <= TIE_TOLERANCE_SECONDS:
            actions.append(Action("skip", rel, "content differs, same modified time"))
        elif delta > 0:
            actions.append(Action("overwrite_local", rel, f"Drive newer by {_human(delta)}", item))
        else:
            actions.append(Action("skip", rel, f"local newer by {_human(-delta)}; never uploaded"))
    return actions


def _human(seconds: float) -> str:
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{seconds / size:.1f}{unit}"
    return f"{seconds:.0f}s"


# ─── Orchestration ───────────────────────────────────────────────────────────


def find_drive_root(service, name: str = DRIVE_ROOT_NAME) -> DriveFile:
    """The existing top-level folder. Never created here: a typo must not fork the tree."""
    root = drive.find_child(service, "root", name)
    if root is None or not root.is_folder:
        raise FileNotFoundError(f"No folder 'My Drive/{name}' on Drive")
    return root


def plan_sync(
    service, repo_root: Path, drive_root_id: str, only: str | None = None
) -> list[Action]:
    groups = [only] if only else list(SCOPE_GROUPS)
    actions: list[Action] = []
    for group in groups:
        planner = plan_push if group == "code" else plan_pull
        for scope in SCOPE_GROUPS[group]:
            logger.info(f"Indexing {scope}")
            actions += planner(
                local_index(repo_root, scope), remote_index(service, drive_root_id, scope)
            )
    return actions


def apply_actions(service, actions: list[Action], repo_root: Path, drive_root_id: str) -> None:
    folder_ids: dict[str, str] = {}

    def parent_id(rel: str) -> str:
        rel_dir = str(PurePosixPath(rel).parent)
        if rel_dir == ".":
            return drive_root_id
        if rel_dir not in folder_ids:
            folder_ids[rel_dir] = drive.ensure_folder_path(service, rel_dir, drive_root_id)
        return folder_ids[rel_dir]

    todo = [a for a in actions if a.kind not in ("unchanged", "skip")]
    for n, action in enumerate(todo, 1):
        local_path = repo_root / action.rel_path
        logger.info(f"[{n}/{len(todo)}] {action.kind} {action.rel_path}")
        if action.kind == "upload":
            drive.upload_file(service, local_path, parent_id(action.rel_path))
        elif action.kind == "update":
            drive.update_file(service, action.remote.id, local_path)
        elif action.kind in ("download", "overwrite_local"):
            drive.download_file(service, action.remote.id, local_path, action.remote.modified_time)
        elif action.kind == "trash":
            drive.delete_file(service, action.remote.id)


# ─── Reports ─────────────────────────────────────────────────────────────────


def format_plan(actions: list[Action]) -> str:
    counts = Counter(a.kind for a in actions)
    lines: list[str] = []
    for kind, label in KIND_LABELS.items():
        group = [a for a in actions if a.kind == kind]
        if not group:
            continue
        lines.append(f"\n{label}: {len(group)}")
        lines.extend(f"    {a.rel_path}  — {a.reason}" for a in group)
    summary = ", ".join(f"{counts[k]} {k}" for k in [*KIND_LABELS, "unchanged"] if counts[k])
    lines.append(f"\nTotal: {summary or 'nothing to sync'}")
    return "\n".join(lines)


def format_listing(service, drive_root_id: str, path: str = "", recursive: bool = False) -> str:
    """``ls -l`` of a folder under the Drive root, for inspecting Drive by hand."""
    folder_id = drive_root_id
    if path:
        node = drive.resolve_path(service, path, drive_root_id)
        if node is None:
            raise FileNotFoundError(f"No {path!r} under the Drive root")
        if not node.is_folder:
            return _listing_line(node, path)
        folder_id = node.id
    lines: list[str] = []
    _list_into(service, folder_id, "", recursive, lines)
    return "\n".join(lines) or "(empty)"


def _list_into(service, folder_id: str, prefix: str, recursive: bool, lines: list[str]) -> None:
    for item in sorted(
        drive.list_folder(service, folder_id), key=lambda f: (not f.is_folder, f.name)
    ):
        lines.append(_listing_line(item, f"{prefix}{item.name}"))
        if recursive and item.is_folder:
            _list_into(service, item.id, f"{prefix}{item.name}/", recursive, lines)


def _listing_line(item: DriveFile, rel: str) -> str:
    size = "<dir>" if item.is_folder else f"{item.size or 0:>10,}"
    stamp = item.modified_time.astimezone().strftime("%Y-%m-%d %H:%M")
    suffix = "/" if item.is_folder else ""
    return f"{size:>12}  {stamp}  {rel}{suffix}"
