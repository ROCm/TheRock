"""Storage backend abstraction for writing to S3 and local directories.

Provides a unified interface for writing files to storage — uploading
local files, copying between storage locations, etc.  Content types for
known file extensions are set automatically during upload.

Usage::

    from _therock_utils.storage_backend import create_storage_backend

    backend = create_storage_backend()  # S3
    backend = create_storage_backend(staging_dir=Path("/tmp/out"))  # local
    backend = create_storage_backend(dry_run=True)  # print only

    # Upload local files
    backend.upload_file(source, dest_location)
    backend.upload_directory(source_dir, dest_location, include=["*.tar.xz*"])

    # Copy between storage locations (e.g. S3-to-S3 promotion)
    backend.copy_file(source_location, dest_location)
"""

import logging
import mimetypes
import os
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path

from _therock_utils.storage_location import StorageLocation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Content-type inference
# ---------------------------------------------------------------------------

# Explicit content-type overrides for extensions we know about.
_CONTENT_TYPE_OVERRIDES: dict[str, str] = {
    ".gz": "application/gzip",
    ".html": "text/html",
    ".log": "text/plain",
    ".md": "text/plain",
    ".whl": "application/zip",
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
# StorageBackend ABC
# ---------------------------------------------------------------------------


class StorageBackend(ABC):
    """Abstract base class for storage operations.

    Subclasses implement the low-level file operations; the ABC provides
    higher-level helpers like ``upload_directory``.
    """

    @abstractmethod
    def upload_file(self, source: Path, dest: StorageLocation) -> None:
        """Upload a single local file to the given destination."""
        ...

    @abstractmethod
    def copy_file(self, source: StorageLocation, dest: StorageLocation) -> None:
        """Copy a file between two storage locations.

        For S3 backends this is a server-side copy (no local download).
        For local backends this copies between local paths.
        """
        ...

    def upload_directory(
        self,
        source_dir: Path,
        dest: StorageLocation,
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
            file_dest = StorageLocation(dest.bucket, f"{dest.relative_path}/{rel}")
            self.upload_file(f, file_dest)
            count += 1
        return count


# ---------------------------------------------------------------------------
# S3StorageBackend
# ---------------------------------------------------------------------------

# Retry parameters for transient S3 errors.
_S3_MAX_RETRIES = 3
_S3_INITIAL_BACKOFF_SECONDS = 1.0


def _s3_retry(operation: str, location: str, func, *args, **kwargs):
    """Call *func* with retries and exponential backoff on failure."""
    last_exc: Exception | None = None
    for attempt in range(_S3_MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < _S3_MAX_RETRIES - 1:
                wait = _S3_INITIAL_BACKOFF_SECONDS * (2**attempt)
                logger.warning(
                    "S3 %s attempt %d/%d failed for %s: %s. Retrying in %.1fs...",
                    operation,
                    attempt + 1,
                    _S3_MAX_RETRIES,
                    location,
                    exc,
                    wait,
                )
                time.sleep(wait)
    raise RuntimeError(
        f"S3 {operation} failed after {_S3_MAX_RETRIES} attempts: {location}"
    ) from last_exc


class S3StorageBackend(StorageBackend):
    """S3 storage backend using boto3.

    The S3 client is lazily initialized on first use.  Credentials are
    resolved through boto3's default credential chain, which checks (in
    order): environment variables, shared credentials file
    (``AWS_SHARED_CREDENTIALS_FILE``), instance metadata, etc.  This
    matches how the ``aws`` CLI resolves credentials.
    """

    def __init__(self, *, dry_run: bool = False):
        self._dry_run = dry_run
        self._s3_client = None

    @property
    def s3_client(self):
        """Lazy-initialized boto3 S3 client."""
        if self._s3_client is None:
            import boto3

            self._s3_client = boto3.client("s3")
        return self._s3_client

    def upload_file(self, source: Path, dest: StorageLocation) -> None:
        content_type = infer_content_type(source)
        if self._dry_run:
            logger.info("[DRY RUN] %s -> %s (%s)", source, dest.s3_uri, content_type)
            return

        _s3_retry(
            "upload",
            dest.s3_uri,
            self.s3_client.upload_file,
            str(source),
            dest.bucket,
            dest.relative_path,
            ExtraArgs={"ContentType": content_type},
        )

    def copy_file(self, source: StorageLocation, dest: StorageLocation) -> None:
        if self._dry_run:
            logger.info("[DRY RUN] copy %s -> %s", source.s3_uri, dest.s3_uri)
            return

        copy_source = {"Bucket": source.bucket, "Key": source.relative_path}
        _s3_retry(
            "copy",
            f"{source.s3_uri} -> {dest.s3_uri}",
            self.s3_client.copy_object,
            Bucket=dest.bucket,
            Key=dest.relative_path,
            CopySource=copy_source,
        )


# ---------------------------------------------------------------------------
# LocalStorageBackend
# ---------------------------------------------------------------------------


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend.

    Mirrors the remote directory layout under *staging_dir* so that
    downstream tools can be tested against a local file tree.
    """

    def __init__(self, staging_dir: Path, *, dry_run: bool = False):
        self._staging_dir = staging_dir
        self._dry_run = dry_run

    def upload_file(self, source: Path, dest: StorageLocation) -> None:
        target = dest.local_path(self._staging_dir)
        if self._dry_run:
            logger.info("[DRY RUN] %s -> %s", source, target)
            return

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    def copy_file(self, source: StorageLocation, dest: StorageLocation) -> None:
        src = source.local_path(self._staging_dir)
        dst = dest.local_path(self._staging_dir)
        if self._dry_run:
            logger.info("[DRY RUN] copy %s -> %s", src, dst)
            return

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_storage_backend(
    *,
    staging_dir: Path | None = None,
    dry_run: bool = False,
) -> StorageBackend:
    """Create a storage backend.

    Args:
        staging_dir: If provided, returns a ``LocalStorageBackend``
            that copies files under this directory.  Otherwise returns an
            ``S3StorageBackend``.
        dry_run: If ``True``, the backend logs actions without writing.
    """
    if staging_dir is not None:
        return LocalStorageBackend(staging_dir, dry_run=dry_run)
    return S3StorageBackend(dry_run=dry_run)
