# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Abstraction layer for artifact storage backends (S3, local directory, or HTTP).

This module provides a unified interface for artifact storage that works with:
- Local directories (for prototyping/testing)
- S3 (for CI/CD)
- HTTP artifact server (read-only access to workflow builds)

TODO(scotttodd): Consolidate with StorageBackend in storage_backend.py? Both
modules manage S3 clients and local directory mirroring. ArtifactBackend has
download/list/exists operations that StorageBackend doesn't have yet.

Environment-based switching:
- THEROCK_HTTP_RUN_ID set → use HTTPBackend (read-only)
- THEROCK_LOCAL_STAGING_DIR set → use LocalDirectoryBackend
- Otherwise → use S3Backend
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set
import hashlib
import os
import re
import shutil
import urllib.request

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


class HTTPBackend(ArtifactBackend):
    """Read-only backend for HTTP artifact server.

    Accesses artifacts from:
    {base_url}/{run_id}-{platform}/

    Artifacts are organized by gfx target via index-{gfx_family}.html files.

    Usage:
        # Create backend for specific run
        backend = HTTPBackend(
            run_id="23309603946",
            platform="linux",
            gfx_families=["gfx94X-dcgpu", "gfx1200"],
            base_url="https://example.com/artifacts"
        )

        # List available artifacts (across all specified GFX families)
        artifacts = backend.list_artifacts(name_filter="blas")

        # Download artifact with checksum verification
        backend.download_artifact("blas_lib_gfx1200.tar.zst", Path("/tmp/blas.tar.zst"))

        # Or use factory function with target families
        import os
        os.environ["THEROCK_RUN_ID"] = "23309603946"
        os.environ["THEROCK_HTTP_BASE_URL"] = "https://example.com/artifacts"
        backend = create_backend_from_env(gfx_families=["gfx94X-dcgpu", "gfx1200"])

    Environment Variables:
        THEROCK_RUN_ID: Workflow run ID (falls back to GITHUB_RUN_ID or "local")
        THEROCK_HTTP_BASE_URL: Base URL for artifact server (required for HTTP backend)
        THEROCK_AMDGPU_FAMILIES: Comma-separated list of GFX families (e.g., "gfx94X-dcgpu,gfx1200")
        THEROCK_PLATFORM: Platform name (default: "linux")
    """

    def __init__(
        self,
        run_id: str,
        base_url: str,
        gfx_families: List[str],
        platform: str = "linux",
    ):
        self.run_id = run_id
        self.base_url = base_url
        self.platform = platform
        self.gfx_families = gfx_families
        self._artifact_cache: Optional[List[str]] = None  # Lazy-loaded artifact list

    @property
    def base_uri(self) -> str:
        """Return the base URI for this backend."""
        return f"{self.base_url}/{self.run_id}-{self.platform}"

    def _download_file(self, url: str, dest: Path) -> None:
        """Download a file from URL to destination path."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            raise FileNotFoundError(f"Failed to download {url}: {e}")

    def _parse_index_html(self, html_content: str) -> List[str]:
        """Parse artifact index HTML to extract artifact filenames."""
        # Extract all href values
        hrefs = re.findall(r'href="([^"]+)"', html_content)

        # Filter to artifact archives only
        artifacts = []
        for href in hrefs:
            # Skip parent directory, index files, http links, anchors
            if href.startswith(("..", "index", "http", "#")):
                continue
            # Only include recognized artifact archives
            if _is_artifact_archive(href):
                artifacts.append(href)

        return artifacts

    def _fetch_index(self, gfx_family: str) -> List[str]:
        """Fetch artifact list from index-{gfx_family}.html."""
        index_url = f"{self.base_uri}/index-{gfx_family}.html"
        try:
            with urllib.request.urlopen(index_url) as response:
                html_content = response.read().decode("utf-8")
            return self._parse_index_html(html_content)
        except Exception:
            # Index doesn't exist for this target
            return []

    def _discover_gfx_families_from_master_index(self) -> List[str]:
        """Discover available GFX families from master index.

        TODO: Future enhancement - Master index discovery
        ==================================================
        This stub will be implemented to fetch and parse the master index at:
        {base_url}/{workflow_id}-linux/index.html

        The master index should contain links to all available index files:
          <a href="../{run_id}-{platform}/index-gfx94X-dcgpu.html">...</a>
          <a href="../{run_id}-{platform}/index-gfx120X-all.html">...</a>

        Implementation plan:
        1. Fetch {base_url}/{workflow_id}-linux/index.html
        2. Parse HTML for links matching pattern: index-{gfx_family}.html
        3. Extract gfx_family from each link
        4. Return list of discovered targets

        This will replace the hardcoded common_targets list and enable
        automatic discovery of all available GFX families.

        Returns:
            List of discovered GFX families (e.g., ["gfx94X-dcgpu", "gfx120X-all", "gfx908"])
        """
        # TODO: Implement master index parsing
        # master_index_url = f"{self.base_url}/{workflow_id}-linux/index.html"
        # try:
        #     with urllib.request.urlopen(master_index_url) as response:
        #         html_content = response.read().decode("utf-8")
        #
        #     # Parse for links like: href="../{run_id}-{platform}/index-{gfx_family}.html"
        #     pattern = rf'href="[^"]*/{self.run_id}-{self.platform}/index-([^"]+)\.html"'
        #     matches = re.findall(pattern, html_content)
        #     return matches
        # except Exception:
        #     # Fall back to hardcoded targets if master index unavailable
        #     return []

        # For now, return empty list to indicate not implemented
        return []

    def list_artifacts(self, name_filter: Optional[str] = None) -> List[str]:
        """List available artifact filenames across all specified GFX families.

        Args:
            name_filter: Optional artifact name prefix to filter by (e.g., "blas" to match "blas_lib_*")

        Returns:
            List of artifact filenames (e.g., ["blas_lib_gfx1200.tar.zst", "rocfft_lib_gfx1200.tar.xz"])
        """
        # Use cache if available
        if self._artifact_cache is None:
            # Fetch from all specified GFX families
            all_artifacts = set()
            for family in self.gfx_families:
                try:
                    artifacts = self._fetch_index(family)
                    all_artifacts.update(artifacts)
                except Exception:
                    # Index doesn't exist for this family - this is expected until
                    # master index is available. Skip silently.
                    continue
            self._artifact_cache = sorted(all_artifacts)

        artifacts = self._artifact_cache

        # Apply name filter if provided
        if name_filter is not None:
            artifacts = [a for a in artifacts if a.startswith(f"{name_filter}_")]

        return sorted(artifacts)

    def _verify_checksum(self, artifact_path: Path) -> bool:
        """Verify artifact SHA256 checksum against .sha256sum file."""
        checksum_file = artifact_path.parent / f"{artifact_path.name}.sha256sum"
        if not checksum_file.exists():
            return False

        # Read expected checksum (first word of .sha256sum file)
        expected = checksum_file.read_text().split()[0]

        # Calculate actual checksum
        sha256 = hashlib.sha256()
        with open(artifact_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        actual = sha256.hexdigest()

        return expected == actual

    def download_artifact(self, artifact_key: str, dest_path: Path) -> None:
        """Download artifact with checksum verification.

        Args:
            artifact_key: The artifact filename (e.g., "blas_lib_gfx1200.tar.zst")
            dest_path: Local path to write the artifact to

        Raises:
            FileNotFoundError: If artifact or checksum not found
            ValueError: If checksum verification fails
        """
        artifact_url = f"{self.base_uri}/{artifact_key}"
        checksum_url = f"{artifact_url}.sha256sum"

        # Download artifact
        self._download_file(artifact_url, dest_path)

        # Download checksum
        checksum_path = dest_path.parent / f"{artifact_key}.sha256sum"
        try:
            self._download_file(checksum_url, checksum_path)

            # Verify checksum
            if not self._verify_checksum(dest_path):
                # Clean up and raise error
                dest_path.unlink(missing_ok=True)
                checksum_path.unlink(missing_ok=True)
                raise ValueError(f"Checksum verification failed for {artifact_key}")

        except FileNotFoundError:
            # Artifacts are allowed to be downloaded without checksums
            pass

    def upload_artifact(self, source_path: Path, artifact_key: str) -> None:
        """Upload is not supported - HTTPBackend is read-only."""
        raise NotImplementedError("HTTPBackend is read-only")

    def copy_artifact(
        self, artifact_key: str, source_backend: "ArtifactBackend"
    ) -> None:
        """Copy is not supported - HTTPBackend is read-only."""
        raise NotImplementedError("HTTPBackend is read-only")

    def artifact_exists(self, artifact_key: str) -> bool:
        """Check if an artifact exists in the backend."""
        # Check if artifact is in cached list (if cache is populated)
        if self._artifact_cache is not None:
            return artifact_key in self._artifact_cache

        # Otherwise, try HTTP HEAD request
        artifact_url = f"{self.base_uri}/{artifact_key}"
        try:
            req = urllib.request.Request(artifact_url, method="HEAD")
            urllib.request.urlopen(req)
            return True
        except Exception:
            return False


def create_backend_from_env(
    run_id: Optional[str] = None,
    platform: Optional[str] = None,
    gfx_families: Optional[List[str]] = None,
) -> ArtifactBackend:
    """Create the appropriate backend based on environment variables.

    Backend priority:
    1. Local directory (THEROCK_LOCAL_STAGING_DIR) - highest priority for local dev
    2. S3 with credentials (AWS_* vars present) - CI upload contexts
    3. HTTP (THEROCK_HTTP_BASE_URL set, no S3 creds) - read-only artifact access
    4. S3 without credentials - fallback for read-only S3 access

    Args:
        run_id: Override run ID (default: THEROCK_RUN_ID or GITHUB_RUN_ID or "local")
        platform: Override platform (default: THEROCK_PLATFORM or system platform)
        gfx_families: List of GFX families for HTTP backend (e.g., ["gfx94X-dcgpu", "gfx1200"])
                    If None, reads from THEROCK_AMDGPU_FAMILIES environment variable (comma-separated)
                    Required for HTTP backend.

    Environment variables:
    - THEROCK_RUN_ID: Workflow run ID (default: GITHUB_RUN_ID or "local")
    - THEROCK_PLATFORM: Override platform (default: current platform)
    - THEROCK_LOCAL_STAGING_DIR: If set, use local backend
    - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN: If all present,
      prioritize S3 backend for upload capability (CI contexts)
    - THEROCK_HTTP_BASE_URL: Base URL for HTTP artifact server
      (if set and no S3 credentials, use HTTP backend for read-only access)
    - THEROCK_AMDGPU_FAMILIES: Comma-separated list of GFX families (e.g., "gfx94X-dcgpu,gfx1200")
      Required for HTTP backend if gfx_families parameter not provided.

    For S3 backend:
    - Uses WorkflowOutputRoot.from_workflow_run() for bucket selection
    """
    import platform as platform_module

    platform_name = platform or os.getenv(
        "THEROCK_PLATFORM", platform_module.system().lower()
    )
    run_id = run_id or os.getenv("THEROCK_RUN_ID", os.getenv("GITHUB_RUN_ID", "local"))

    # Priority 1: Local directory backend (for local development)
    local_staging = os.getenv("THEROCK_LOCAL_STAGING_DIR")
    if local_staging:
        output_root = WorkflowOutputRoot.for_local(
            run_id=run_id, platform=platform_name
        )
        return LocalDirectoryBackend(
            staging_dir=Path(local_staging),
            output_root=output_root,
        )

    # Priority 2: S3 backend when AWS credentials are present (CI upload contexts)
    # Check for all three required AWS credentials
    has_s3_credentials = all(
        os.getenv(var)
        for var in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"]
    )
    if has_s3_credentials:
        output_root = WorkflowOutputRoot.from_workflow_run(
            run_id=run_id, platform=platform_name
        )
        return S3Backend(output_root=output_root)

    # Priority 3: HTTP backend for read-only access (no upload capability)
    # Only use HTTP backend if base URL is explicitly configured
    http_base_url = os.getenv("THEROCK_HTTP_BASE_URL")
    if http_base_url:
        # Get GFX families from parameter or environment variable
        targets = gfx_families
        if targets is None:
            targets_env = os.getenv("THEROCK_AMDGPU_FAMILIES")
            if targets_env:
                targets = [t.strip() for t in targets_env.split(",")]

        if not targets:
            raise ValueError(
                "HTTPBackend requires gfx_families. "
                "Provide via gfx_families parameter or THEROCK_AMDGPU_FAMILIES environment variable "
                "(comma-separated list, e.g., 'gfx94X-dcgpu,gfx1200')"
            )

        return HTTPBackend(
            run_id=run_id,
            base_url=http_base_url,
            gfx_families=targets,
            platform=platform_name,
        )

    # Priority 4: S3 backend without credentials (read-only fallback)
    output_root = WorkflowOutputRoot.from_workflow_run(
        run_id=run_id, platform=platform_name
    )
    return S3Backend(output_root=output_root)
