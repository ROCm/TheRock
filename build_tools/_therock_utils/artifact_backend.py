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
from pathlib import Path
from typing import List, Optional, Set
import os
import shutil
import tempfile
from urllib.error import HTTPError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, url2pathname, urlopen

from .workflow_outputs import WorkflowOutputRoot
from .storage_location import StorageLocation


@dataclass
class ArtifactLocation:
    """Represents an artifact's location in the backend."""

    artifact_key: str  # e.g., "blas_lib_gfx94X.tar.zst" or "blas_lib_gfx94X.tar.xz"
    full_path: str  # Backend-specific full path/URI


# Supported artifact archive extensions (in order of preference)
ARTIFACT_EXTENSIONS = (".tar.zst", ".tar.xz")
HTTP_DOWNLOAD_TIMEOUT_SECONDS = 120


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
    def upload_artifact(self, source_path: Path, artifact_key: str) -> None:
        """Upload/copy a local artifact to the backend.

        Args:
            source_path: Local path of the artifact to upload
            artifact_key: The artifact filename to use in the backend
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
        with tempfile.TemporaryDirectory(prefix="therock-artifact-copy-") as temporary:
            archive_path = Path(temporary) / artifact_key
            source_backend.download_artifact(artifact_key, archive_path)
            self.upload_artifact(archive_path, artifact_key)
            checksum_key = f"{artifact_key}.sha256sum"
            if source_backend.artifact_exists(checksum_key):
                checksum_path = Path(temporary) / checksum_key
                source_backend.download_artifact(checksum_key, checksum_path)
                self.upload_artifact(checksum_path, checksum_key)

    @property
    @abstractmethod
    def base_uri(self) -> str:
        """Return the base URI/path for this backend."""
        pass


class LocalDirectoryBackend(ArtifactBackend):
    """Backend using a local directory (for testing/prototyping).

    Without output_root, staging_dir is the exact artifact directory.
    Otherwise, directory structure mirrors S3 layout via WorkflowOutputRoot::

        {staging_dir}/{output_root.prefix}/
            {artifact_name}_{component}_{target_family}.tar.zst
    """

    def __init__(
        self, staging_dir: Path, output_root: WorkflowOutputRoot | None = None
    ):
        self.staging_dir = Path(staging_dir)
        self.output_root = output_root
        self.base_path.mkdir(parents=True, exist_ok=True)

    @property
    def base_path(self) -> Path:
        """Local artifacts directory path."""
        return (
            self.staging_dir / self.output_root.prefix
            if self.output_root is not None
            else self.staging_dir
        )

    @property
    def base_uri(self) -> str:
        return str(self.base_path)

    def _artifact_path(self, artifact_key: str) -> Path:
        """Get local path for an artifact file."""
        return self.base_path / artifact_key

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

    def upload_artifact(self, source_path: Path, artifact_key: str) -> None:
        """Copy artifact from source to staging."""
        if not source_path.exists():
            raise FileNotFoundError(f"Source artifact not found: {source_path}")
        dest = self._artifact_path(artifact_key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest)
        # Also copy sha256sum if it exists
        sha_src = source_path.parent / f"{source_path.name}.sha256sum"
        if sha_src.exists():
            shutil.copy2(sha_src, self._artifact_path(f"{artifact_key}.sha256sum"))

    def copy_artifact(
        self, artifact_key: str, source_backend: "ArtifactBackend"
    ) -> None:
        """Copy artifact from another local backend."""
        if not isinstance(source_backend, LocalDirectoryBackend):
            return super().copy_artifact(artifact_key, source_backend)
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

    def __init__(self, output_root: WorkflowOutputRoot | StorageLocation):
        self.output_root = output_root
        self.location = (
            output_root.root()
            if isinstance(output_root, WorkflowOutputRoot)
            else output_root
        )
        self._s3_client = None

    @property
    def bucket(self) -> str:
        return self.location.bucket

    @property
    def s3_prefix(self) -> str:
        return self.location.relative_path

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
        return self.location.s3_uri

    def _artifact_location(self, artifact_key: str) -> StorageLocation:
        """Resolve a filename under an explicit root or a workflow output root."""
        key = f"{self.s3_prefix}/{artifact_key}".lstrip("/")
        return StorageLocation(self.bucket, key)

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
        loc = self._artifact_location(artifact_key)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        self.s3_client.download_file(self.bucket, loc.relative_path, str(dest_path))

    def upload_artifact(self, source_path: Path, artifact_key: str) -> None:
        """Upload to S3."""
        loc = self._artifact_location(artifact_key)
        self.s3_client.upload_file(str(source_path), self.bucket, loc.relative_path)

    def copy_artifact(
        self, artifact_key: str, source_backend: "ArtifactBackend"
    ) -> None:
        """Server-side copy from another S3 backend (cross-bucket supported)."""
        if not isinstance(source_backend, S3Backend):
            return super().copy_artifact(artifact_key, source_backend)
        copy_source = {
            "Bucket": source_backend.bucket,
            "Key": source_backend._artifact_location(artifact_key).relative_path,
        }
        dest_key = self._artifact_location(artifact_key).relative_path
        self.s3_client.copy(copy_source, self.bucket, dest_key)
        # Also copy sha256sum if it exists
        sha_key = f"{artifact_key}.sha256sum"
        if source_backend.artifact_exists(sha_key):
            sha_copy_source = {
                "Bucket": source_backend.bucket,
                "Key": source_backend._artifact_location(sha_key).relative_path,
            }
            self.s3_client.copy(
                sha_copy_source,
                self.bucket,
                self._artifact_location(sha_key).relative_path,
            )

    def artifact_exists(self, artifact_key: str) -> bool:
        """Check if artifact exists in S3."""
        try:
            loc = self._artifact_location(artifact_key)
            self.s3_client.head_object(Bucket=self.bucket, Key=loc.relative_path)
            return True
        except Exception:
            return False


class HTTPSBackend(ArtifactBackend):
    """Read explicitly named files from an HTTPS artifact root."""

    def __init__(self, root: str):
        self.root = root.rstrip("/")

    @property
    def base_uri(self) -> str:
        return self.root

    def download_artifact(self, artifact_key: str, dest_path: Path) -> None:
        """Stream one file to disk; propagate missing files and network failures."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(
            f"{self.root}/{artifact_key}", timeout=HTTP_DOWNLOAD_TIMEOUT_SECONDS
        ) as response:
            with dest_path.open("wb") as output:
                shutil.copyfileobj(response, output)

    def artifact_exists(self, artifact_key: str) -> bool:
        """Check an optional sidecar; only HTTP 404 means it is absent."""
        try:
            with urlopen(
                Request(f"{self.root}/{artifact_key}", method="HEAD"),
                timeout=HTTP_DOWNLOAD_TIMEOUT_SECONDS,
            ):
                return True
        except HTTPError as error:
            if error.code == 404:
                return False
            raise

    def list_artifacts(self, name_filter: Optional[str] = None) -> List[str]:
        raise NotImplementedError("HTTPS sources require explicit filenames")

    def upload_artifact(self, source_path: Path, artifact_key: str) -> None:
        raise NotImplementedError("HTTPS artifact roots are read-only")

    def copy_artifact(self, artifact_key: str, source_backend: ArtifactBackend) -> None:
        raise NotImplementedError("HTTPS artifact roots are read-only")


def create_backend_from_uri(root: str) -> ArtifactBackend:
    """Open a recorded source root without inferring a producer run or bucket."""
    location = urlsplit(root)
    if location.query or location.fragment or location.username or location.password:
        raise ValueError(
            "Artifact roots must not contain credentials, queries, or fragments"
        )
    if location.scheme == "file":
        if location.netloc not in ("", "localhost"):
            raise ValueError("Only local file URLs are supported")
        directory = Path(url2pathname(location.path))
        if not directory.is_absolute() or not directory.is_dir():
            raise ValueError(f"Artifact source directory does not exist: {directory}")
        return LocalDirectoryBackend(staging_dir=directory.resolve())
    if location.scheme == "s3" and location.netloc:
        return S3Backend(StorageLocation(location.netloc, location.path.strip("/")))
    if location.scheme == "https" and location.netloc:
        if location.hostname.endswith(".s3.amazonaws.com") and location.port in (
            None,
            443,
        ):
            bucket = location.hostname.removesuffix(".s3.amazonaws.com")
            return S3Backend(StorageLocation(bucket, unquote(location.path).strip("/")))
        return HTTPSBackend(root)
    raise ValueError(
        "Artifact source must be an s3://, https://, or local file:// root"
    )


def create_backend_from_env(
    run_id: Optional[str] = None,
    github_repository: Optional[str] = None,
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
        run_id=run_id, platform=platform_name, github_repository=github_repository
    )
    return S3Backend(output_root=output_root)
