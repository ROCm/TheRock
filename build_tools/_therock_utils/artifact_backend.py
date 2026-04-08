# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Abstraction layer for artifact storage backends (S3 or local directory).

This module provides a unified interface for artifact storage that works with
both local directories (for prototyping/testing) and S3 (for CI/CD).

TODO(scotttodd): Consolidate with StorageBackend in storage_backend.py? Both
modules manage S3 clients and local directory mirroring. ArtifactBackend has
download/list/exists operations that StorageBackend doesn't have yet.

Environment-based switching:
- THEROCK_LOCAL_STAGING_DIR set → use LocalDirectoryBackend
- Otherwise → use S3Backend
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import List, Optional
import os
import shutil

from .hash_util import calculate_hash
from .workflow_outputs import WorkflowOutputRoot

logger = logging.getLogger(__name__)


class ArtifactConflictError(Exception):
    """Raised when an artifact would overwrite an existing one with different content.

    When uploading an artifact that already exists, the SHA256 checksums are compared.
    If they differ, it indicates a potential build system problem - either the artifact
    should have a different name/version, or an unexpected change occurred.
    """

    pass


def _parse_sha256sum_content(content: str) -> str:
    """Parse a sha256sum file and return the hash.

    The format is: {hash}  {filename}
    (two spaces between hash and filename)

    Args:
        content: The content of the sha256sum file.

    Returns:
        The SHA256 hash from the file.

    Raises:
        ValueError: If the content cannot be parsed.
    """
    content = content.strip()
    if not content:
        raise ValueError("Empty sha256sum content")
    # Format is: hash  filename (two spaces)
    parts = content.split("  ", 1)
    if len(parts) < 1 or not parts[0]:
        raise ValueError(f"Invalid sha256sum format: {content!r}")
    return parts[0].strip()


def _read_local_sha256sum(source_path: Path) -> Optional[str]:
    """Read the SHA256 hash from a local artifact's companion .sha256sum file.

    Args:
        source_path: Path to the artifact file.

    Returns:
        The SHA256 hash if the sha256sum file exists, None otherwise.
    """
    sha_path = source_path.parent / f"{source_path.name}.sha256sum"
    if not sha_path.exists():
        return None
    try:
        return _parse_sha256sum_content(sha_path.read_text())
    except (ValueError, OSError) as e:
        logger.warning("Failed to read local sha256sum %s: %s", sha_path, e)
        return None


def _compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file.

    Args:
        file_path: Path to the file.

    Returns:
        Hexadecimal SHA256 hash string.
    """
    return calculate_hash(file_path, "sha256").hexdigest()


@dataclass
class ArtifactLocation:
    """Represents an artifact's location in the backend."""

    artifact_key: str  # e.g., "blas_lib_gfx94X.tar.zst" or "blas_lib_gfx94X.tar.xz"
    full_path: str  # Backend-specific full path/URI


# Supported artifact archive extensions (in order of preference)
ARTIFACT_EXTENSIONS = (".tar.zst", ".tar.xz")


def _is_artifact_archive(filename: str) -> bool:
    """Check if a filename is a recognized artifact archive."""
    return any(filename.endswith(ext) for ext in ARTIFACT_EXTENSIONS)


class ArtifactBackend(ABC):
    """Abstract base for artifact storage backends."""

    @abstractmethod
    def list_artifacts(self, name_filter: Optional[str] = None) -> List[str]:
        """List available artifact filenames.

        Args:
            name_filter: Optional artifact name prefix to filter by (e.g., "blas" to match "blas_lib_*")

        Returns:
            List of artifact filenames (e.g., ["blas_lib_gfx94X.tar.zst", "blas_dev_gfx94X.tar.xz"])
        """
        pass

    @abstractmethod
    def download_artifact(self, artifact_key: str, dest_path: Path) -> None:
        """Download/copy an artifact to a local path.

        Args:
            artifact_key: The artifact filename (e.g., "blas_lib_gfx94X.tar.xz")
            dest_path: Local path to write the artifact to
        """
        pass

    @abstractmethod
    def upload_artifact(
        self,
        source_path: Path,
        artifact_key: str,
        sha256sum_path: Path | None = None,
    ) -> None:
        """Upload/copy a local artifact to the backend.

        Implementations must perform conflict detection: if an artifact already
        exists with different content, raise ``ArtifactConflictError``. If it
        exists with identical content, skip the upload. This prevents silent
        overwrites that could mask build system problems.

        If sha256sum_path is provided and exists, it is also uploaded as a
        companion file (artifact_key + ".sha256sum").

        Args:
            source_path: Local path of the artifact to upload.
            artifact_key: The artifact filename to use in the backend.
            sha256sum_path: Optional path to local sha256sum file. If provided,
                used for efficient conflict detection without computing hash,
                and also uploaded alongside the artifact.

        Raises:
            ArtifactConflictError: If the artifact already exists with
                different content (different SHA256 or file contents).
        """
        pass

    @abstractmethod
    def artifact_exists(self, artifact_key: str) -> bool:
        """Check if an artifact exists in the backend."""
        pass

    @abstractmethod
    def copy_artifact(
        self, artifact_key: str, source_backend: "ArtifactBackend"
    ) -> None:
        """Copy an artifact from source_backend into this backend (server-side when possible).

        Also copies the companion .sha256sum file if it exists in the source.

        Args:
            artifact_key: The artifact filename (e.g., "blas_lib_gfx94X.tar.zst")
            source_backend: The backend to copy from
        """
        pass

    @property
    @abstractmethod
    def base_uri(self) -> str:
        """Return the base URI/path for this backend."""
        pass


class LocalDirectoryBackend(ArtifactBackend):
    """Backend using a local directory (for testing/prototyping).

    Directory structure mirrors S3 layout via WorkflowOutputRoot::

        {staging_dir}/{output_root.prefix}/
            {artifact_name}_{component}_{target_family}.tar.zst
    """

    def __init__(self, staging_dir: Path, output_root: WorkflowOutputRoot):
        self.staging_dir = Path(staging_dir)
        self.output_root = output_root
        self.base_path.mkdir(parents=True, exist_ok=True)

    @property
    def base_path(self) -> Path:
        """Local artifacts directory path."""
        return self.staging_dir / self.output_root.prefix

    @property
    def base_uri(self) -> str:
        return str(self.base_path)

    def _artifact_path(self, artifact_key: str) -> Path:
        """Get local path for an artifact file."""
        return self.output_root.artifact(artifact_key).local_path(self.staging_dir)

    def list_artifacts(self, name_filter: Optional[str] = None) -> List[str]:
        """List artifacts in local staging directory."""
        artifacts = []
        if not self.base_path.exists():
            return artifacts
        for p in self.base_path.iterdir():
            filename = p.name
            # Skip non-artifact files (also excludes .sha256sum files)
            if not _is_artifact_archive(filename):
                continue
            # Apply name filter if provided
            if name_filter is not None and not filename.startswith(f"{name_filter}_"):
                continue
            artifacts.append(filename)
        return sorted(artifacts)

    def download_artifact(self, artifact_key: str, dest_path: Path) -> None:
        """Copy artifact from staging to destination."""
        src = self._artifact_path(artifact_key)
        if not src.exists():
            raise FileNotFoundError(f"Artifact not found in local staging: {src}")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_path)
        # Also copy sha256sum if it exists
        sha_src = self._artifact_path(f"{artifact_key}.sha256sum")
        if sha_src.exists():
            shutil.copy2(sha_src, dest_path.parent / f"{artifact_key}.sha256sum")

    def upload_artifact(
        self,
        source_path: Path,
        artifact_key: str,
        sha256sum_path: Path | None = None,
    ) -> None:
        """Copy artifact from source to staging.

        Performs SHA256 conflict detection to prevent silent overwrites of
        existing artifacts with different content. If the artifact already
        exists with the same SHA256, the upload is skipped.

        Args:
            source_path: Local path of the artifact to upload.
            artifact_key: The artifact filename to use in the backend.
            sha256sum_path: Optional path to local sha256sum file for efficient
                conflict detection.

        Raises:
            FileNotFoundError: If source_path does not exist.
            ArtifactConflictError: If the artifact already exists with a
                different SHA256 checksum.
        """
        if not source_path.exists():
            raise FileNotFoundError(f"Source artifact not found: {source_path}")

        # Check for SHA conflicts if artifact already exists
        dest = self._artifact_path(artifact_key)
        if dest.exists():
            existing_sha_path = self._artifact_path(f"{artifact_key}.sha256sum")
            existing_sha: Optional[str] = None

            # Try to get existing SHA from .sha256sum file
            if existing_sha_path.exists():
                try:
                    existing_sha = _parse_sha256sum_content(
                        existing_sha_path.read_text()
                    )
                except (ValueError, OSError) as e:
                    logger.warning(
                        "Failed to read existing sha256sum for %s: %s. "
                        "Will compute from file contents.",
                        artifact_key,
                        e,
                    )

            # If no sha256sum file, compute hash from the existing artifact
            if existing_sha is None:
                logger.info(
                    "Artifact %s exists but has no sha256sum file. "
                    "Computing hash from file contents.",
                    artifact_key,
                )
                existing_sha = _compute_file_sha256(dest)

            # Get local SHA: prefer provided path, then sibling, then compute
            local_sha: Optional[str] = None
            if sha256sum_path is not None and sha256sum_path.exists():
                try:
                    local_sha = _parse_sha256sum_content(sha256sum_path.read_text())
                except (ValueError, OSError) as e:
                    logger.warning(
                        "Failed to read provided sha256sum %s: %s",
                        sha256sum_path,
                        e,
                    )
            if local_sha is None:
                local_sha = _read_local_sha256sum(source_path)
            if local_sha is None:
                local_sha = _compute_file_sha256(source_path)

            if existing_sha == local_sha:
                logger.info(
                    "Skipping upload of %s: identical artifact "
                    "already exists (SHA256: %s)",
                    artifact_key,
                    existing_sha[:16] + "...",
                )
                return
            else:
                raise ArtifactConflictError(
                    f"Artifact {artifact_key} already exists "
                    f"with different content.\n"
                    f"  Existing SHA256: {existing_sha}\n"
                    f"  Local SHA256:    {local_sha}\n"
                    f"This indicates a potential build system problem."
                )

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest)
        # Also copy sha256sum if provided
        if sha256sum_path is not None and sha256sum_path.exists():
            shutil.copy2(
                sha256sum_path, self._artifact_path(f"{artifact_key}.sha256sum")
            )

    def copy_artifact(
        self, artifact_key: str, source_backend: "ArtifactBackend"
    ) -> None:
        """Copy artifact from another local backend."""
        if not isinstance(source_backend, LocalDirectoryBackend):
            raise TypeError(
                f"Cannot copy from {type(source_backend).__name__} to LocalDirectoryBackend"
            )
        src = source_backend.base_path / artifact_key
        if not src.exists():
            raise FileNotFoundError(f"Artifact not found in source backend: {src}")
        dest = self.base_path / artifact_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        # Also copy sha256sum if it exists
        sha_src = source_backend.base_path / f"{artifact_key}.sha256sum"
        if sha_src.exists():
            shutil.copy2(sha_src, self.base_path / f"{artifact_key}.sha256sum")

    def artifact_exists(self, artifact_key: str) -> bool:
        """Check if artifact exists in local staging."""
        return self._artifact_path(artifact_key).exists()


class S3Backend(ArtifactBackend):
    """Backend using AWS S3.

    S3 path structure is defined by WorkflowOutputRoot::

        s3://{bucket}/{prefix}/
            {artifact_name}_{component}_{target_family}.tar.zst
    """

    def __init__(self, output_root: WorkflowOutputRoot):
        self.output_root = output_root
        self._s3_client = None

    @property
    def bucket(self) -> str:
        return self.output_root.bucket

    @property
    def s3_prefix(self) -> str:
        return self.output_root.prefix

    @property
    def s3_client(self):
        """Lazy-initialized boto3 S3 client.

        Credentials are resolved through boto3's default credential chain
        (see https://docs.aws.amazon.com/boto3/latest/guide/credentials.html).
        Relevant locations are checked in order:

        1. Environment variables (``AWS_ACCESS_KEY_ID``,
           ``AWS_SECRET_ACCESS_KEY``, ``AWS_SESSION_TOKEN``)
        2. Assume role providers
        3. Shared credentials file (``AWS_SHARED_CREDENTIALS_FILE``)

        When no credentials are found at all, the client falls back to
        unsigned requests for public bucket reads.
        """
        if self._s3_client is None:
            import boto3
            from botocore import UNSIGNED
            from botocore.config import Config

            session = boto3.Session()
            credentials = session.get_credentials()

            if credentials is not None:
                self._s3_client = session.client(
                    "s3",
                    verify=True,
                    config=Config(max_pool_connections=100),
                )
            else:
                self._s3_client = session.client(
                    "s3",
                    verify=True,
                    config=Config(max_pool_connections=100, signature_version=UNSIGNED),
                )
        return self._s3_client

    @property
    def base_uri(self) -> str:
        return self.output_root.root().s3_uri

    def _fetch_sha256sum(self, sha_key: str) -> str:
        """Fetch and parse a sha256sum file from S3.

        Args:
            sha_key: The sha256sum filename (e.g., 'artifact.tar.zst.sha256sum')

        Returns:
            The SHA256 hash from the file.

        Raises:
            ValueError: If the content cannot be parsed.
            Exception: If the S3 download fails.
        """
        import io

        loc = self.output_root.artifact(sha_key)
        buffer = io.BytesIO()
        self.s3_client.download_fileobj(self.bucket, loc.relative_path, buffer)
        content = buffer.getvalue().decode("utf-8")
        return _parse_sha256sum_content(content)

    def _compute_remote_sha256(self, artifact_key: str) -> str:
        """Download artifact from S3 and compute its SHA256 hash.

        This is used when the .sha256sum file is missing and we need to
        compare file contents directly.

        Args:
            artifact_key: The artifact filename.

        Returns:
            Hexadecimal SHA256 hash string.
        """
        import tempfile

        loc = self.output_root.artifact(artifact_key)
        # Use delete=False to allow S3 client to write to the file on Windows
        # (NamedTemporaryFile with delete=True keeps the file open exclusively)
        tmp = tempfile.NamedTemporaryFile(delete=False)
        try:
            tmp.close()  # Close so S3 client can write to it
            self.s3_client.download_file(self.bucket, loc.relative_path, tmp.name)
            return _compute_file_sha256(Path(tmp.name))
        finally:
            os.unlink(tmp.name)

    def list_artifacts(self, name_filter: Optional[str] = None) -> List[str]:
        """List S3 artifacts."""
        paginator = self.s3_client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=self.bucket, Prefix=self.s3_prefix)

        artifacts = []
        for page in page_iterator:
            if "Contents" not in page:
                continue
            for obj in page["Contents"]:
                key = obj["Key"]
                # Extract filename from full key
                if "/" in key:
                    filename = key.split("/")[-1]
                else:
                    filename = key
                # Skip non-artifact files (also excludes .sha256sum files)
                if not _is_artifact_archive(filename):
                    continue
                # Apply name filter if provided
                if name_filter is not None and not filename.startswith(
                    f"{name_filter}_"
                ):
                    continue
                artifacts.append(filename)
        return sorted(set(artifacts))

    def download_artifact(self, artifact_key: str, dest_path: Path) -> None:
        """Download from S3."""
        loc = self.output_root.artifact(artifact_key)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        self.s3_client.download_file(self.bucket, loc.relative_path, str(dest_path))

    def upload_artifact(
        self,
        source_path: Path,
        artifact_key: str,
        sha256sum_path: Path | None = None,
    ) -> None:
        """Upload artifact to S3.

        Performs SHA256 conflict detection to prevent silent overwrites of
        existing artifacts with different content. If the artifact already
        exists with the same SHA256, the upload is skipped.

        Args:
            source_path: Local path of the artifact to upload.
            artifact_key: The artifact filename to use in the backend.
            sha256sum_path: Optional path to local sha256sum file for efficient
                conflict detection.

        Raises:
            ArtifactConflictError: If the artifact already exists with a
                different SHA256 checksum.
            RuntimeError: If there's a network error during the existence check.
        """
        # Check for SHA conflicts if artifact already exists
        if self.artifact_exists(artifact_key):
            # Get existing SHA: prefer remote sha256sum file, fall back to download
            existing_sha: Optional[str] = None
            sha_key = f"{artifact_key}.sha256sum"
            if self.artifact_exists(sha_key):
                try:
                    existing_sha = self._fetch_sha256sum(sha_key)
                except (ValueError, OSError) as e:
                    logger.warning(
                        "Failed to fetch existing sha256sum for %s: %s. "
                        "Will compute from file contents.",
                        artifact_key,
                        e,
                    )

            if existing_sha is None:
                # No sha256sum file or failed to read - download and compute
                logger.info(
                    "Artifact %s exists but has no readable sha256sum file. "
                    "Computing hash from file contents.",
                    artifact_key,
                )
                existing_sha = self._compute_remote_sha256(artifact_key)

            # Get local SHA: prefer provided path, then sibling, then compute
            local_sha: Optional[str] = None
            if sha256sum_path is not None and sha256sum_path.exists():
                try:
                    local_sha = _parse_sha256sum_content(sha256sum_path.read_text())
                except (ValueError, OSError) as e:
                    logger.warning(
                        "Failed to read provided sha256sum %s: %s",
                        sha256sum_path,
                        e,
                    )
            if local_sha is None:
                local_sha = _read_local_sha256sum(source_path)
            if local_sha is None:
                local_sha = _compute_file_sha256(source_path)

            if existing_sha == local_sha:
                logger.info(
                    "Skipping upload of %s: identical artifact "
                    "already exists in S3 (SHA256: %s)",
                    artifact_key,
                    existing_sha[:16] + "...",
                )
                return
            else:
                raise ArtifactConflictError(
                    f"Artifact {artifact_key} already exists "
                    f"in S3 with different content.\n"
                    f"  Existing SHA256: {existing_sha}\n"
                    f"  Local SHA256:    {local_sha}\n"
                    f"  S3 location:     s3://{self.bucket}/"
                    f"{self.output_root.artifact(artifact_key).relative_path}\n"
                    f"This indicates a potential build system problem."
                )

        loc = self.output_root.artifact(artifact_key)
        self.s3_client.upload_file(str(source_path), self.bucket, loc.relative_path)
        # Also upload sha256sum if provided
        if sha256sum_path is not None and sha256sum_path.exists():
            sha_loc = self.output_root.artifact(f"{artifact_key}.sha256sum")
            self.s3_client.upload_file(
                str(sha256sum_path), self.bucket, sha_loc.relative_path
            )

    def copy_artifact(
        self, artifact_key: str, source_backend: "ArtifactBackend"
    ) -> None:
        """Server-side copy from another S3 backend (cross-bucket supported)."""
        if not isinstance(source_backend, S3Backend):
            raise TypeError(
                f"Cannot copy from {type(source_backend).__name__} to S3Backend"
            )
        copy_source = {
            "Bucket": source_backend.bucket,
            "Key": f"{source_backend.s3_prefix}/{artifact_key}",
        }
        dest_key = f"{self.s3_prefix}/{artifact_key}"
        self.s3_client.copy(copy_source, self.bucket, dest_key)
        # Also copy sha256sum if it exists
        sha_key = f"{artifact_key}.sha256sum"
        if source_backend.artifact_exists(sha_key):
            sha_copy_source = {
                "Bucket": source_backend.bucket,
                "Key": f"{source_backend.s3_prefix}/{sha_key}",
            }
            self.s3_client.copy(
                sha_copy_source, self.bucket, f"{self.s3_prefix}/{sha_key}"
            )

    def artifact_exists(self, artifact_key: str) -> bool:
        """Check if artifact exists in S3."""
        try:
            loc = self.output_root.artifact(artifact_key)
            self.s3_client.head_object(Bucket=self.bucket, Key=loc.relative_path)
            return True
        except Exception:
            return False


def create_backend_from_env(
    run_id: Optional[str] = None,
    platform: Optional[str] = None,
) -> ArtifactBackend:
    """Create the appropriate backend based on environment variables.

    Environment variables:
    - THEROCK_LOCAL_STAGING_DIR: If set, use local backend
    - THEROCK_RUN_ID: Override run ID (default: "local" or GITHUB_RUN_ID)
    - THEROCK_PLATFORM: Override platform (default: current platform)

    For S3 backend (when THEROCK_LOCAL_STAGING_DIR is not set):
    - Uses WorkflowOutputRoot.from_workflow_run() for bucket selection
    """
    import platform as platform_module

    local_staging = os.getenv("THEROCK_LOCAL_STAGING_DIR")
    platform_name = platform or os.getenv(
        "THEROCK_PLATFORM", platform_module.system().lower()
    )
    run_id = run_id or os.getenv("THEROCK_RUN_ID", os.getenv("GITHUB_RUN_ID", "local"))

    if local_staging:
        output_root = WorkflowOutputRoot.for_local(
            run_id=run_id, platform=platform_name
        )
        return LocalDirectoryBackend(
            staging_dir=Path(local_staging),
            output_root=output_root,
        )

    output_root = WorkflowOutputRoot.from_workflow_run(
        run_id=run_id, platform=platform_name
    )
    return S3Backend(output_root=output_root)
