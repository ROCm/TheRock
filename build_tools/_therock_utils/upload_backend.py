"""Upload backend abstraction for CI run outputs.

Provides a unified interface for uploading files and directories to S3 or
local staging directories. Content types for known file extensions are set
during upload.

Usage::

    from _therock_utils.upload_backend import create_upload_backend

    backend = create_upload_backend()  # S3
    backend = create_upload_backend(staging_dir=Path("/tmp/out"))  # local
    backend = create_upload_backend(dry_run=True)  # print only

    backend.upload_file(source, dest_location)
    backend.upload_directory(source_dir, dest_location, include=["*.tar.xz*"])
"""

import logging
import mimetypes
import os
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path

from _therock_utils.run_outputs import OutputLocation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Content-type inference
# ---------------------------------------------------------------------------

# Explicit content-type overrides for extensions we know about.
_CONTENT_TYPE_OVERRIDES: dict[str, str] = {
    ".gz": "application/gzip",
    ".log": "text/plain",
    ".md": "text/plain",
    ".xz": "application/x-xz",
    ".zst": "application/zstd",
}

_DEFAULT_CONTENT_TYPE = "application/octet-stream"


def infer_content_type(path: Path) -> str:
    """Infer MIME content-type from a file's extension.

    Uses explicit overrides for extensions we know about, falling back
    to ``mimetypes`` for everything else.
    """
    suffix = path.suffix.lower()
    if suffix in _CONTENT_TYPE_OVERRIDES:
        return _CONTENT_TYPE_OVERRIDES[suffix]
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or _DEFAULT_CONTENT_TYPE


# ---------------------------------------------------------------------------
# UploadBackend ABC
# ---------------------------------------------------------------------------


class UploadBackend(ABC):
    """Abstract base class for uploading files to a storage backend."""

    @abstractmethod
    def upload_file(self, source: Path, dest: OutputLocation) -> None:
        """Upload a single file to the given destination."""
        ...

    def upload_directory(
        self,
        source_dir: Path,
        dest: OutputLocation,
        include: list[str] | None = None,
    ) -> int:
        """Upload files from *source_dir* to *dest*, preserving relative paths.

        Args:
            source_dir: Local directory to upload from.
            dest: Destination location (the directory root in the backend).
            include: Optional glob patterns to filter files (e.g.
                ``["*.tar.xz*"]``). If ``None``, all files are uploaded.

        Returns:
            Number of files uploaded.

        Symlinks are skipped. Subdirectory structure is preserved.
        """
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        patterns = include or ["*"]
        files: set[Path] = set()
        for pattern in patterns:
            files.update(source_dir.rglob(pattern))
        sorted_files = sorted(f for f in files if f.is_file() and not f.is_symlink())

        count = 0
        for f in sorted_files:
            rel = f.relative_to(source_dir).as_posix()
            file_dest = OutputLocation(dest.bucket, f"{dest.relative_path}/{rel}")
            self.upload_file(f, file_dest)
            count += 1
        return count


# ---------------------------------------------------------------------------
# S3UploadBackend
# ---------------------------------------------------------------------------

# Retry parameters for transient S3 errors.
_S3_MAX_RETRIES = 3
_S3_INITIAL_BACKOFF_SECONDS = 1.0


class S3UploadBackend(UploadBackend):
    """Upload files to AWS S3 using boto3.

    The S3 client is lazily initialized on first use.  If
    ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, and
    ``AWS_SESSION_TOKEN`` are all set, an authenticated client is created;
    otherwise an unsigned client is used (matching the pattern in
    ``artifact_backend.S3Backend``).
    """

    def __init__(self, *, dry_run: bool = False):
        self._dry_run = dry_run
        self._s3_client = None

    @property
    def s3_client(self):
        """Lazy-initialized boto3 S3 client."""
        if self._s3_client is None:
            import boto3

            access_key = os.environ.get("AWS_ACCESS_KEY_ID")
            secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
            session_token = os.environ.get("AWS_SESSION_TOKEN")

            if None not in (access_key, secret_key, session_token):
                self._s3_client = boto3.client(
                    "s3",
                    verify=True,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    aws_session_token=session_token,
                )
            else:
                from botocore import UNSIGNED
                from botocore.config import Config

                self._s3_client = boto3.client(
                    "s3",
                    verify=True,
                    config=Config(signature_version=UNSIGNED),
                )
        return self._s3_client

    def upload_file(self, source: Path, dest: OutputLocation) -> None:
        content_type = infer_content_type(source)
        if self._dry_run:
            logger.info("[DRY RUN] %s -> %s (%s)", source, dest.s3_uri, content_type)
            return

        last_exc: Exception | None = None
        for attempt in range(_S3_MAX_RETRIES):
            try:
                self.s3_client.upload_file(
                    str(source),
                    dest.bucket,
                    dest.relative_path,
                    ExtraArgs={"ContentType": content_type},
                )
                return
            except Exception as exc:
                last_exc = exc
                if attempt < _S3_MAX_RETRIES - 1:
                    wait = _S3_INITIAL_BACKOFF_SECONDS * (2**attempt)
                    logger.warning(
                        "S3 upload attempt %d/%d failed for %s: %s. Retrying in %.1fs...",
                        attempt + 1,
                        _S3_MAX_RETRIES,
                        dest.s3_uri,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
        raise RuntimeError(
            f"S3 upload failed after {_S3_MAX_RETRIES} attempts: {dest.s3_uri}"
        ) from last_exc


# ---------------------------------------------------------------------------
# LocalUploadBackend
# ---------------------------------------------------------------------------


class LocalUploadBackend(UploadBackend):
    """Copy files to a local staging directory.

    Mirrors the remote directory layout under *staging_dir* so that
    downstream tools can be tested against a local file tree.
    """

    def __init__(self, staging_dir: Path, *, dry_run: bool = False):
        self._staging_dir = staging_dir
        self._dry_run = dry_run

    def upload_file(self, source: Path, dest: OutputLocation) -> None:
        target = dest.local_path(self._staging_dir)
        if self._dry_run:
            logger.info("[DRY RUN] %s -> %s", source, target)
            return

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_upload_backend(
    *,
    staging_dir: Path | None = None,
    dry_run: bool = False,
) -> UploadBackend:
    """Create an upload backend.

    Args:
        staging_dir: If provided, returns a `LocalUploadBackend`
            that copies files under this directory.  Otherwise returns an
            `S3UploadBackend`.
        dry_run: If ``True``, the backend logs actions without writing.
    """
    if staging_dir is not None:
        return LocalUploadBackend(staging_dir, dry_run=dry_run)
    return S3UploadBackend(dry_run=dry_run)
