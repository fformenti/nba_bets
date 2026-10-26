"""Small Google Drive API primitives: list, find, create, upload, download, delete.

No sync knowledge lives here — see ``src/gdrive/sync.py`` for that. Every
function takes the ``service`` returned by :func:`get_service` as its first
argument, so nothing talks to Google at import time and the module imports
without credentials (``tests/test_imports_smoke.py``).

Auth is the OAuth desktop-app flow with the full ``drive`` scope: the narrower
``drive.file`` scope only sees files this app created, and the ``nba_bets``
folder on Drive was created by hand and by Colab.
"""

import hashlib
import io
import mimetypes
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload, MediaIoBaseUpload

from src.config.secrets import (
    DRIVE_CLIENT_SECRETS_ENV_VAR,
    DRIVE_CLIENT_SECRETS_FILE,
    DRIVE_TOKEN_ENV_VAR,
    DRIVE_TOKEN_FILE,
)

SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_MIME = "application/vnd.google-apps.folder"
GOOGLE_NATIVE_PREFIX = "application/vnd.google-apps."
FILE_FIELDS = "id, name, mimeType, md5Checksum, size, modifiedTime"
# Small files go up in one request; resumable uploads only pay off above this.
RESUMABLE_THRESHOLD_BYTES = 5 * 1024 * 1024
# googleapiclient retries timeouts, 5xx and 429 with exponential backoff, but
# only when asked: its default is 0, so one dropped read killed a 500-file sync.
RETRIES = 5


@dataclass(frozen=True)
class DriveFile:
    id: str
    name: str
    mime_type: str
    md5: str | None
    size: int | None
    modified_time: datetime

    @property
    def is_folder(self) -> bool:
        return self.mime_type == FOLDER_MIME

    @property
    def is_google_native(self) -> bool:
        """Docs, Sheets, shortcuts: no bytes, no md5 — never downloaded or overwritten."""
        return self.mime_type.startswith(GOOGLE_NATIVE_PREFIX) and not self.is_folder

    @classmethod
    def from_api(cls, item: dict) -> "DriveFile":
        return cls(
            id=item["id"],
            name=item["name"],
            mime_type=item["mimeType"],
            md5=item.get("md5Checksum"),
            size=int(item["size"]) if "size" in item else None,
            modified_time=parse_rfc3339(item["modifiedTime"]),
        )


class DuplicateNameError(RuntimeError):
    """Drive allows two children with one name; a path lookup cannot pick one."""


def parse_rfc3339(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def to_rfc3339(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def md5_of(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 — compared with Drive's md5Checksum, not security
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ─── Auth ────────────────────────────────────────────────────────────────────


def _credential_paths(client_secrets: Path | None, token_path: Path | None) -> tuple[Path, Path]:
    secrets = client_secrets or Path(
        os.environ.get(DRIVE_CLIENT_SECRETS_ENV_VAR, DRIVE_CLIENT_SECRETS_FILE)
    )
    token = token_path or Path(os.environ.get(DRIVE_TOKEN_ENV_VAR, DRIVE_TOKEN_FILE))
    return secrets.expanduser(), token.expanduser()


def _save_token(creds: Credentials, token: Path) -> None:
    token.parent.mkdir(parents=True, exist_ok=True)
    token.write_text(creds.to_json())
    token.chmod(0o600)


def run_login(client_secrets: Path | None = None, token_path: Path | None = None) -> Path:
    """Open the browser consent screen once and store the refreshable token."""
    secrets, token = _credential_paths(client_secrets, token_path)
    if not secrets.exists():
        raise FileNotFoundError(
            f"No OAuth client file at {secrets}. Download a 'Desktop app' client JSON from "
            f"Google Cloud (see README, 'Google Drive sync'), or set {DRIVE_CLIENT_SECRETS_ENV_VAR}."
        )
    flow = InstalledAppFlow.from_client_secrets_file(str(secrets), SCOPES)
    creds = flow.run_local_server(port=0)
    _save_token(creds, token)
    return token


def get_service(client_secrets: Path | None = None, token_path: Path | None = None):
    """Drive v3 client from the stored token, refreshed if expired."""
    _, token = _credential_paths(client_secrets, token_path)
    if not token.exists():
        raise RuntimeError(f"No Drive token at {token}. Run `make drive-auth` first.")
    creds = Credentials.from_authorized_user_file(str(token), SCOPES)
    if not creds.valid:
        if not (creds.expired and creds.refresh_token):
            raise RuntimeError(f"Drive token at {token} is unusable. Run `make drive-auth`.")
        creds.refresh(Request())
        _save_token(creds, token)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


# ─── Read ────────────────────────────────────────────────────────────────────


def list_folder(service, folder_id: str) -> list[DriveFile]:
    """Direct children of a folder, trashed items excluded."""
    files: list[DriveFile] = []
    page_token = None
    while True:
        response = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields=f"nextPageToken, files({FILE_FIELDS})",
                pageSize=1000,
                pageToken=page_token,
            )
            .execute(num_retries=RETRIES)
        )
        files.extend(DriveFile.from_api(item) for item in response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            return files


def find_child(service, parent_id: str, name: str) -> DriveFile | None:
    escaped = name.replace("\\", "\\\\").replace("'", "\\'")
    response = (
        service.files()
        .list(
            q=f"'{parent_id}' in parents and name = '{escaped}' and trashed = false",
            fields=f"files({FILE_FIELDS})",
        )
        .execute(num_retries=RETRIES)
    )
    matches = response.get("files", [])
    if len(matches) > 1:
        raise DuplicateNameError(f"{len(matches)} items named {name!r} in folder {parent_id}")
    return DriveFile.from_api(matches[0]) if matches else None


def resolve_path(service, path: str, root_id: str = "root") -> DriveFile | None:
    """Follow a slash-separated path down from ``root_id``; None if any part is missing."""
    current: DriveFile | None = None
    parent_id = root_id
    for part in [p for p in path.split("/") if p]:
        current = find_child(service, parent_id, part)
        if current is None:
            return None
        parent_id = current.id
    return current


def walk_folder(service, folder_id: str, prefix: str = "") -> dict[str, list[DriveFile]]:
    """Every file below a folder, keyed by posix path relative to it.

    Values are lists because Drive permits duplicate names; callers decide what
    a duplicate means. Folders themselves are not keys — only their contents.
    """
    found: dict[str, list[DriveFile]] = {}
    for item in list_folder(service, folder_id):
        rel = f"{prefix}{item.name}"
        if item.is_folder:
            for sub_rel, sub_items in walk_folder(service, item.id, f"{rel}/").items():
                found.setdefault(sub_rel, []).extend(sub_items)
        else:
            found.setdefault(rel, []).append(item)
    return found


def download_file(service, file_id: str, dest: Path, modified_time: datetime | None = None) -> None:
    """Write a file's bytes to ``dest``; stamp ``modified_time`` as its local mtime."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.part")
    with tmp.open("wb") as handle:
        downloader = MediaIoBaseDownload(handle, service.files().get_media(fileId=file_id))
        done = False
        while not done:
            _, done = downloader.next_chunk(num_retries=RETRIES)
    tmp.replace(dest)
    if modified_time is not None:
        stamp = modified_time.timestamp()
        os.utime(dest, (stamp, stamp))


def read_text(service, file_id: str, encoding: str = "utf-8") -> str:
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, service.files().get_media(fileId=file_id))
    done = False
    while not done:
        _, done = downloader.next_chunk(num_retries=RETRIES)
    return buffer.getvalue().decode(encoding)


# ─── Write ───────────────────────────────────────────────────────────────────


def get_or_create_folder(service, name: str, parent_id: str) -> str:
    existing = find_child(service, parent_id, name)
    if existing is not None:
        if not existing.is_folder:
            raise NotADirectoryError(f"{name!r} in {parent_id} is a file, not a folder")
        return existing.id
    created = (
        service.files()
        .create(body={"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]}, fields="id")
        .execute(num_retries=RETRIES)
    )
    return created["id"]


def ensure_folder_path(service, rel_dir: str, root_id: str) -> str:
    """``mkdir -p`` below ``root_id``; returns the id of the deepest folder."""
    folder_id = root_id
    for part in [p for p in rel_dir.split("/") if p]:
        folder_id = get_or_create_folder(service, part, folder_id)
    return folder_id


def _media(local_path: Path) -> MediaFileUpload:
    mime = mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
    resumable = local_path.stat().st_size > RESUMABLE_THRESHOLD_BYTES
    return MediaFileUpload(str(local_path), mimetype=mime, resumable=resumable)


def _local_mtime(local_path: Path) -> str:
    return to_rfc3339(datetime.fromtimestamp(local_path.stat().st_mtime).astimezone())


def upload_file(service, local_path: Path, parent_id: str, name: str | None = None) -> DriveFile:
    """Create a new Drive file; its modifiedTime is the local file's mtime."""
    body = {
        "name": name or local_path.name,
        "parents": [parent_id],
        "modifiedTime": _local_mtime(local_path),
    }
    item = (
        service.files()
        .create(body=body, media_body=_media(local_path), fields=FILE_FIELDS)
        .execute(num_retries=RETRIES)
    )
    return DriveFile.from_api(item)


def update_file(service, file_id: str, local_path: Path) -> DriveFile:
    """Replace a file's content in place: same id, old content kept as a revision."""
    item = (
        service.files()
        .update(
            fileId=file_id,
            body={"modifiedTime": _local_mtime(local_path)},
            media_body=_media(local_path),
            fields=FILE_FIELDS,
        )
        .execute(num_retries=RETRIES)
    )
    return DriveFile.from_api(item)


def write_text(service, parent_id: str, name: str, text: str, encoding: str = "utf-8") -> DriveFile:
    """Create or overwrite a small text file by name."""
    media = MediaIoBaseUpload(
        io.BytesIO(text.encode(encoding)),
        mimetype=mimetypes.guess_type(name)[0] or "text/plain",
    )
    existing = find_child(service, parent_id, name)
    if existing is not None:
        item = (
            service.files()
            .update(fileId=existing.id, media_body=media, fields=FILE_FIELDS)
            .execute(num_retries=RETRIES)
        )
    else:
        item = (
            service.files()
            .create(
                body={"name": name, "parents": [parent_id]}, media_body=media, fields=FILE_FIELDS
            )
            .execute(num_retries=RETRIES)
        )
    return DriveFile.from_api(item)


def delete_file(service, file_id: str, *, permanent: bool = False) -> None:
    """Move to Drive's trash (recoverable for 30 days), or delete outright."""
    if permanent:
        service.files().delete(fileId=file_id).execute(num_retries=RETRIES)
    else:
        service.files().update(fileId=file_id, body={"trashed": True}).execute(num_retries=RETRIES)
