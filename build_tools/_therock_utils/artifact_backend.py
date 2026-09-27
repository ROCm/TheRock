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

from .workflow_outputs import WorkflowOutputRoot


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

    def upload_artifact(self, source_path: Path, artifact_key: str) -> None:
        """Upload to S3."""
        loc = self.output_root.artifact(artifact_key)
        self.s3_client.upload_file(str(source_path), self.bucket, loc.relative_path)

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


class ShimS3Backend(S3Backend):
    """S3Backend that DOWNLOADS via a node-local Dragonfly s3-shim (plain HTTP
    GET) instead of boto3, so repeat artifact pulls serve P2P from the mesh and
    cut direct S3 egress. Only download_artifact is redirected; list/exists/
    upload/copy stay on boto3 (metadata + CI-publish paths, not egress-heavy).

    The shim 302s the artifact GET to the node dfdaemon, which back-sources the
    real object via a proxy rule, so the request path must be the PATH-STYLE
    object path ``/{bucket}/{key}`` (per-run buckets work behind one shim). Reads
    are already anonymous against a public bucket, so nothing is lost over HTTP.
    Enabled by THEROCK_ARTIFACT_HTTP_BASE (e.g. http://$NODE_IP:4011).
    """

    def __init__(self, output_root: WorkflowOutputRoot, shim_base: str):
        super().__init__(output_root=output_root)
        self._shim_base = shim_base.rstrip("/")

    def download_artifact(self, artifact_key: str, dest_path: Path) -> None:
        """Download via the shim (HTTP GET), atomically, with length check."""
        import urllib.error
        import urllib.request

        loc = self.output_root.artifact(artifact_key)
        url = f"{self._shim_base}/{self.bucket}/{loc.relative_path}"
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest_path.with_name(dest_path.name + ".part")
        try:
            with urllib.request.urlopen(url, timeout=600) as resp:
                expected = resp.headers.get("Content-Length")
                written = 0
                with open(tmp, "wb") as f:
                    while chunk := resp.read(1 << 20):
                        f.write(chunk)
                        written += len(chunk)
            # Guard against a silent short read poisoning the download cache
            # (the caller treats an existing file as "cached — skip").
            if expected is not None and written != int(expected):
                raise IOError(
                    f"short read {written}/{expected} bytes for {artifact_key}"
                )
            os.replace(tmp, dest_path)
        except urllib.error.HTTPError as e:
            tmp.unlink(missing_ok=True)
            if e.code == 404:
                # Match S3/skip semantics: caller treats FileNotFoundError as skip.
                raise FileNotFoundError(artifact_key) from e
            raise
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise


def s3_backend_from_env(output_root: WorkflowOutputRoot) -> S3Backend:
    """Return a ShimS3Backend when THEROCK_ARTIFACT_HTTP_BASE is set (route
    downloads through the node-local Dragonfly s3-shim), else a plain S3Backend.
    Off the shim env, behavior is byte-identical to before."""
    shim_base = os.getenv("THEROCK_ARTIFACT_HTTP_BASE")
    if shim_base:
        return ShimS3Backend(output_root=output_root, shim_base=shim_base)
    return S3Backend(output_root=output_root)


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
    return s3_backend_from_env(output_root)
