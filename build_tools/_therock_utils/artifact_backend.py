"""Abstraction layer for artifact storage backends (S3 or local directory).

This module provides a unified interface for artifact storage that works with
both local directories (for prototyping/testing) and S3 (for CI/CD).

Environment-based switching:
- THEROCK_LOCAL_STAGING_DIR set → use LocalDirectoryBackend
- Otherwise → use S3Backend with existing retrieve_bucket_info() logic

Path computation is delegated to RunOutputRoot (see run_outputs.py) to ensure
consistent paths across all tools.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set
import os
import shutil

from .run_outputs import RunOutputRoot


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

    @property
    @abstractmethod
    def base_uri(self) -> str:
        """Return the base URI/path for this backend."""
        pass


class LocalDirectoryBackend(ArtifactBackend):
    """Backend using a local directory (for testing/prototyping).

    Directory structure mirrors S3:
        {staging_dir}/{run_id}-{platform}/
            {artifact_name}_{component}_{target_family}.tar.zst
            {artifact_name}_{component}_{target_family}.tar.zst.sha256sum

    Uses RunOutputRoot for path computation to ensure consistency with S3 layout.
    """

    def __init__(self, staging_dir: Path, run_root: RunOutputRoot):
        self.staging_dir = Path(staging_dir)
        self.run_root = run_root
        self.base_path = run_root.local_path(self.staging_dir)
        self.base_path.mkdir(parents=True, exist_ok=True)

    @property
    def base_uri(self) -> str:
        return str(self.base_path)

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
        src = self.base_path / artifact_key
        if not src.exists():
            raise FileNotFoundError(f"Artifact not found in local staging: {src}")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_path)
        # Also copy sha256sum if it exists
        sha_src = self.base_path / f"{artifact_key}.sha256sum"
        if sha_src.exists():
            shutil.copy2(sha_src, dest_path.parent / f"{artifact_key}.sha256sum")

    def upload_artifact(self, source_path: Path, artifact_key: str) -> None:
        """Copy artifact from source to staging."""
        if not source_path.exists():
            raise FileNotFoundError(f"Source artifact not found: {source_path}")
        dest = self.base_path / artifact_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest)
        # Also copy sha256sum if it exists
        sha_src = source_path.parent / f"{source_path.name}.sha256sum"
        if sha_src.exists():
            shutil.copy2(sha_src, self.base_path / f"{artifact_key}.sha256sum")

    def artifact_exists(self, artifact_key: str) -> bool:
        """Check if artifact exists in local staging."""
        return (self.base_path / artifact_key).exists()


class S3Backend(ArtifactBackend):
    """Backend using AWS S3.

    S3 path structure:
        s3://{bucket}/{prefix}/
            {artifact_name}_{component}_{target_family}.tar.zst

    Uses RunOutputRoot for path computation to ensure consistency across tools.
    """

    def __init__(self, run_root: RunOutputRoot):
        self.run_root = run_root
        self._s3_client = None

    @property
    def base_uri(self) -> str:
        return self.run_root.s3_uri

    @property
    def bucket(self) -> str:
        return self.run_root.bucket

    @property
    def s3_prefix(self) -> str:
        return self.run_root.prefix

    @property
    def s3_client(self):
        """Lazy initialization of S3 client."""
        if self._s3_client is None:
            import boto3
            from botocore import UNSIGNED
            from botocore.config import Config

            _access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
            _secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
            _session_token = os.environ.get("AWS_SESSION_TOKEN")

            if None not in (_access_key_id, _secret_access_key, _session_token):
                self._s3_client = boto3.client(
                    "s3",
                    verify=True,
                    aws_access_key_id=_access_key_id,
                    aws_secret_access_key=_secret_access_key,
                    aws_session_token=_session_token,
                )
            else:
                self._s3_client = boto3.client(
                    "s3",
                    verify=True,
                    config=Config(max_pool_connections=100, signature_version=UNSIGNED),
                )
        return self._s3_client

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
        s3_key = f"{self.s3_prefix}/{artifact_key}"
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        self.s3_client.download_file(self.bucket, s3_key, str(dest_path))

    def upload_artifact(self, source_path: Path, artifact_key: str) -> None:
        """Upload to S3."""
        s3_key = f"{self.s3_prefix}/{artifact_key}"
        self.s3_client.upload_file(str(source_path), self.bucket, s3_key)

    def artifact_exists(self, artifact_key: str) -> bool:
        """Check if artifact exists in S3."""
        try:
            s3_key = f"{self.s3_prefix}/{artifact_key}"
            self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
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
    - Uses RunOutputRoot.from_workflow_run() for bucket selection
    - Respects GITHUB_REPOSITORY, IS_PR_FROM_FORK, etc.
    """
    import platform as platform_module

    local_staging = os.getenv("THEROCK_LOCAL_STAGING_DIR")
    platform_name = platform or os.getenv(
        "THEROCK_PLATFORM", platform_module.system().lower()
    )
    run_id = run_id or os.getenv("THEROCK_RUN_ID", os.getenv("GITHUB_RUN_ID", "local"))

    if local_staging:
        run_root = RunOutputRoot.for_local(run_id=run_id, platform=platform_name)
        return LocalDirectoryBackend(
            staging_dir=Path(local_staging),
            run_root=run_root,
        )

    # Default to S3
    run_root = RunOutputRoot.from_workflow_run(run_id=run_id, platform=platform_name)
    return S3Backend(run_root=run_root)
