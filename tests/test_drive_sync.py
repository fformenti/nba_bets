"""Drive sync planning: which way each file moves, and what is never touched.

The planner is pure, so these run without credentials or network. The rules
that matter most are the destructive ones: push trashes only inside its scope,
and outputs/runs/ never deletes on either side.
"""

from datetime import UTC, datetime, timedelta

from src.gdrive.drive import FOLDER_MIME, DriveFile
from src.gdrive.sync import LocalFile, is_ignored, local_index, plan_bidirectional, plan_push

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


def local(md5: str, mtime: datetime = T0) -> LocalFile:
    return LocalFile(md5=md5, size=1, mtime=mtime)


def remote(md5: str | None, mtime: datetime = T0, mime: str = "text/plain", name: str = "f"):
    return [
        DriveFile(
            id=f"id-{name}-{md5}", name=name, mime_type=mime, md5=md5, size=1, modified_time=mtime
        )
    ]


def kinds(actions) -> dict[str, str]:
    return {a.rel_path: a.kind for a in actions}


def test_push_uploads_new_updates_changed_and_trashes_extra():
    actions = plan_push(
        {"src/new.py": local("a"), "src/same.py": local("b"), "src/changed.py": local("c")},
        {"src/same.py": remote("b"), "src/changed.py": remote("old"), "src/gone.py": remote("d")},
    )
    assert kinds(actions) == {
        "src/new.py": "upload",
        "src/same.py": "unchanged",
        "src/changed.py": "update",
        "src/gone.py": "trash",
    }


def test_push_ignores_local_junk_and_trashes_drive_junk(tmp_path):
    (tmp_path / "src" / "__pycache__").mkdir(parents=True)
    (tmp_path / "src" / "a.py").write_text("x")
    (tmp_path / "src" / ".DS_Store").write_text("x")
    (tmp_path / "src" / "__pycache__" / "a.cpython-311.pyc").write_text("x")
    assert set(local_index(tmp_path, "src")) == {"src/a.py"}

    actions = plan_push({}, {"src/.DS_Store": remote("j")})
    assert [(a.kind, a.reason) for a in actions] == [("trash", "macOS/Python junk")]


def test_push_never_touches_google_native_or_duplicates():
    doc = remote(None, mime="application/vnd.google-apps.document")
    dupes = remote("a") + remote("b")
    actions = plan_push({"src/x.py": local("a")}, {"src/notes": doc, "src/x.py": dupes})
    assert kinds(actions) == {"src/notes": "skip", "src/x.py": "skip"}


def test_file_versus_folder_clash_is_skipped_not_trashed():
    actions = plan_push({"configs/x": local("a")}, {"configs/x/inner.yaml": remote("b")})
    assert {a.kind for a in actions} == {"skip"}


def test_runs_only_on_drive_downloads_and_only_local_uploads():
    actions = plan_bidirectional(
        {"outputs/runs/local/run.json": local("a")},
        {"outputs/runs/colab/run.json": remote("b")},
    )
    assert kinds(actions) == {
        "outputs/runs/local/run.json": "upload",
        "outputs/runs/colab/run.json": "download",
    }


def test_runs_never_delete_on_either_side():
    actions = plan_bidirectional(
        {"outputs/runs/a/x.csv": local("a")},
        {"outputs/runs/b/y.csv": remote("b"), "outputs/runs/b/.DS_Store": remote("j")},
    )
    assert "trash" not in {a.kind for a in actions}
    assert "outputs/runs/b/.DS_Store" not in kinds(actions)


def test_runs_conflict_newer_side_wins():
    later = T0 + timedelta(hours=1)
    drive_newer = plan_bidirectional({"r/x": local("a", T0)}, {"r/x": remote("b", later)})
    local_newer = plan_bidirectional({"r/x": local("a", later)}, {"r/x": remote("b", T0)})
    assert kinds(drive_newer) == {"r/x": "overwrite_local"}
    assert kinds(local_newer) == {"r/x": "overwrite_drive"}


def test_runs_conflict_with_equal_times_is_skipped():
    actions = plan_bidirectional(
        {"r/x": local("a", T0)}, {"r/x": remote("b", T0 + timedelta(seconds=1))}
    )
    assert kinds(actions) == {"r/x": "skip"}


def test_runs_same_md5_is_unchanged_whatever_the_times():
    actions = plan_bidirectional(
        {"r/x": local("a", T0)}, {"r/x": remote("a", T0 + timedelta(days=3))}
    )
    assert kinds(actions) == {"r/x": "unchanged"}


def test_ignore_rules():
    assert is_ignored("src/.DS_Store")
    assert is_ignored("src/ml/__pycache__/x.py")
    assert is_ignored("colab_notebooks/.ipynb_checkpoints/a.ipynb")
    assert not is_ignored("src/ml/x.py")
    assert FOLDER_MIME.endswith("folder")
