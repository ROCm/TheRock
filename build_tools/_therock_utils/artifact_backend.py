# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Abstraction layer for artifact storage backends (S3, HTTP, or local directory).

This module provides a unified interface for artifact storage that works with
local directories (for prototyping/testing), S3 (for CI/CD), and plain HTTPS
(read-only, no credentials).

TODO(scotttodd): Consolidate with StorageBackend in storage_backend.py? Both
modules manage S3 clients and local directory mirroring. ArtifactBackend has
download/list/exists operations that StorageBackend doesn't have yet.

Backend selection is explicit via ``create_backend(transport=...)``. The
``transport="auto"`` default (also what ``create_backend_from_env`` uses)
resolves to LocalDirectoryBackend when THEROCK_LOCAL_STAGING_DIR is set and to
S3Backend otherwise -- the writable backend for the current run.
"""

from abc import ABC, abstractmethod
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional
from urllib.parse import unquote, urlparse
import os
import shutil
import urllib.error
import urllib.request

from .hash_util import calculate_hash
from .workflow_outputs import WorkflowOutputRoot


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
        # TODO: pass Delimiter="/" and skip CommonPrefixes. Without it this walks
        # the whole subtree and reports e.g. logs/<stage>/ccache_logs.tar.zst as a
        # run-root artifact named "ccache_logs.tar.zst", which then 404s on
        # download. HTTPBackend does not have this bug because it reads only the
        # run root's index. Left alone here because narrowing the listing changes
        # behavior for every S3 caller and belongs in its own change.
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
            # TODO: Narrow this to ClientError with a 404/403 status. A 500 from
            # S3 currently reads as "the artifact does not exist", the same bug
            # HTTPBackend.artifact_exists below avoids.
            return False


class _IndexLinkParser(HTMLParser):
    """Collects href values from an artifact index page.

    A real parser rather than a regex: the generator emits attributes we do not
    control the quoting of, and a regex that silently matches nothing turns a
    broken page into "no artifacts found", which is indistinguishable from a run
    that produced none.
    """

    def __init__(self):
        super().__init__()
        self.hrefs: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value:
                self.hrefs.append(value)


class HTTPBackend(ArtifactBackend):
    """Read-only backend that fetches artifacts over plain HTTPS.

    No credentials and no boto3: this reads the same public URLs a browser
    would. Every URL comes from ``StorageLocation.public_url``, so it resolves
    to a CDN for buckets that have one configured (see
    ``_therock_utils/s3_buckets.py``) and to the raw S3 URL otherwise.

    ``list_artifacts`` reads the generated directory index at
    ``{prefix}/index.html``. Artifacts live directly at the run root, so a
    single fetch enumerates all of them.

    Uploads are not supported; use ``transport='s3'`` for those.

    Usage::

        output_root = WorkflowOutputRoot.from_workflow_run(
            run_id="23309603946", platform="linux"
        )
        backend = HTTPBackend(output_root)
        artifacts = backend.list_artifacts(name_filter="blas")
        backend.download_artifact("blas_lib_gfx1200.tar.zst", Path("/tmp/blas.tar.zst"))
    """

    def __init__(self, output_root: WorkflowOutputRoot, *, timeout: float = 60.0):
        self.output_root = output_root
        self.timeout = timeout
        self._artifact_cache: Optional[List[str]] = None

    @property
    def base_uri(self) -> str:
        return self.output_root.root().public_url

    def _download_file(self, url: str, dest: Path) -> None:
        """Download a URL to a local path.

        Raises:
            FileNotFoundError: The resource does not exist (HTTP 404 or 403).
            ConnectionError: Network-level failure (timeout, DNS, TLS).
            urllib.error.HTTPError: Any other HTTP status, including 5xx.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                with open(dest, "wb") as f:
                    shutil.copyfileobj(response, f)
        except urllib.error.HTTPError as e:
            # HTTPError subclasses URLError, so it has to be caught first.
            # 403 is included because S3 returns it instead of 404 for a missing
            # key when the caller lacks ListBucket.
            if e.code in (403, 404):
                raise FileNotFoundError(f"Not found: {url}") from e
            raise
        except urllib.error.URLError as e:
            dest.unlink(missing_ok=True)
            raise ConnectionError(f"Failed to download {url}: {e}") from e

    def _parse_index_html(self, html_content: str) -> List[str]:
        """Extract artifact filenames from an index page."""
        parser = _IndexLinkParser()
        parser.feed(html_content)

        artifacts = []
        for href in parser.hrefs:
            # The generator percent-encodes hrefs; the filename on disk is the
            # decoded form.
            filename = unquote(href)
            # Skip the parent link, nested index pages, and absolute links.
            if filename.startswith(("..", "index", "#")) or urlparse(filename).scheme:
                continue
            if _is_artifact_archive(filename):
                artifacts.append(filename)
        return artifacts

    def _fetch_index(self) -> List[str]:
        """Fetch and parse the run's artifact index.

        Raises:
            FileNotFoundError: The index does not exist. This is an error rather
                than an empty result: an index is written for every run that
                uploaded anything, so a missing one means the run ID, platform,
                or bucket is wrong, and returning [] would report that as
                "this run has no artifacts".
            ConnectionError: Network-level failure.
            urllib.error.HTTPError: Any other HTTP status, including 5xx.
        """
        index_url = self.output_root.artifact_index().public_url
        try:
            with urllib.request.urlopen(index_url, timeout=self.timeout) as response:
                html_content = response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                raise FileNotFoundError(
                    f"No artifact index at {index_url}. Check the run ID, platform, "
                    f"and bucket."
                ) from e
            raise
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to fetch {index_url}: {e}") from e
        return self._parse_index_html(html_content)

    def list_artifacts(self, name_filter: Optional[str] = None) -> List[str]:
        """List artifacts published for this run."""
        if self._artifact_cache is None:
            self._artifact_cache = sorted(set(self._fetch_index()))

        artifacts = self._artifact_cache
        if name_filter is not None:
            artifacts = [a for a in artifacts if a.startswith(f"{name_filter}_")]
        return sorted(artifacts)

    def download_artifact(self, artifact_key: str, dest_path: Path) -> None:
        """Download an artifact, verifying its sha256 if one is published.

        Raises:
            FileNotFoundError: The artifact does not exist.
            ValueError: The published checksum does not match what was received.
        """
        artifact_url = self.output_root.artifact(artifact_key).public_url
        self._download_file(artifact_url, dest_path)

        checksum_key = f"{artifact_key}.sha256sum"
        checksum_url = self.output_root.artifact(checksum_key).public_url
        checksum_path = dest_path.parent / checksum_key
        try:
            self._download_file(checksum_url, checksum_path)
        except FileNotFoundError:
            # Artifacts may be published without a checksum. Scope this except
            # to the download alone: a FileNotFoundError raised while verifying
            # would otherwise be swallowed as "no checksum published".
            print(f"WARNING: no checksum published for {artifact_key}, not verifying")
            return

        expected = checksum_path.read_text().split()[0]
        actual = calculate_hash(dest_path, "sha256").hexdigest()
        if expected != actual:
            dest_path.unlink(missing_ok=True)
            checksum_path.unlink(missing_ok=True)
            raise ValueError(
                f"Checksum mismatch for {artifact_key} downloaded from "
                f"{artifact_url}: expected {expected}, got {actual}"
            )

    def upload_artifact(self, source_path: Path, artifact_key: str) -> None:
        raise NotImplementedError(
            "HTTPBackend is read-only; use transport='s3' to upload artifacts"
        )

    def copy_artifact(
        self, artifact_key: str, source_backend: "ArtifactBackend"
    ) -> None:
        raise NotImplementedError(
            "HTTPBackend is read-only; use transport='s3' to copy artifacts"
        )

    def artifact_exists(self, artifact_key: str) -> bool:
        """Check whether an artifact is published for this run."""
        # The cache only holds archives, so it cannot answer for a .sha256sum
        # (which S3Backend.copy_artifact asks about). Fall through to a HEAD for
        # anything the index would not have listed.
        if self._artifact_cache is not None and _is_artifact_archive(artifact_key):
            return artifact_key in self._artifact_cache

        artifact_url = self.output_root.artifact(artifact_key).public_url
        request = urllib.request.Request(artifact_url, method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout):
                return True
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return False
            # A 5xx means the server could not answer, not that the object is
            # absent. Reporting "does not exist" here would let a caller
            # silently skip an artifact that is actually present.
            raise
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to probe {artifact_url}: {e}") from e


# Transports accepted by create_backend. "auto" resolves to the writable
# backend for the current run: local staging if configured, S3 otherwise.
TRANSPORTS = ("auto", "s3", "http", "local")

# Transports that can be written to. "http" is read-only.
WRITABLE_TRANSPORTS = ("auto", "s3", "local")


def create_backend(
    run_id: Optional[str] = None,
    github_repository: Optional[str] = None,
    platform: Optional[str] = None,
    *,
    transport: str = "auto",
    staging_dir: Optional[Path] = None,
) -> ArtifactBackend:
    """Create an artifact backend for the given transport.

    Transport is an explicit choice rather than something inferred from ambient
    state, because a single process can need two different backends at once:
    ``artifact_manager copy`` builds a source and a destination, and no
    process-wide setting can express that.

    Args:
        run_id: Run ID (default: THEROCK_RUN_ID, GITHUB_RUN_ID, or "local").
        github_repository: Repository owning the run, for bucket selection.
        platform: Platform name (default: THEROCK_PLATFORM or the current one).
        transport: One of ``TRANSPORTS``.

            * ``auto``  - local staging if THEROCK_LOCAL_STAGING_DIR (or
              ``staging_dir``) is set, otherwise S3. The writable backend for
              the current run.
            * ``s3``    - S3 with boto3 credentials. Read/write.
            * ``http``  - public HTTPS, no credentials. Read-only.
            * ``local`` - a local staging directory. Read/write.
        staging_dir: Staging directory for ``local`` (and for ``auto`` when set).
            Defaults to THEROCK_LOCAL_STAGING_DIR.

    Raises:
        ValueError: Unknown transport, or ``local`` with no staging directory.
    """
    import platform as platform_module

    if transport not in TRANSPORTS:
        raise ValueError(
            f"Unknown transport {transport!r}, expected one of {', '.join(TRANSPORTS)}"
        )

    platform_name = platform or os.getenv(
        "THEROCK_PLATFORM", platform_module.system().lower()
    )
    run_id = run_id or os.getenv("THEROCK_RUN_ID", os.getenv("GITHUB_RUN_ID", "local"))
    staging = staging_dir or os.getenv("THEROCK_LOCAL_STAGING_DIR")

    if transport == "auto":
        transport = "local" if staging else "s3"

    if transport == "local":
        if not staging:
            raise ValueError(
                "transport='local' needs a staging directory; pass staging_dir or "
                "set THEROCK_LOCAL_STAGING_DIR"
            )
        return LocalDirectoryBackend(
            staging_dir=Path(staging),
            output_root=WorkflowOutputRoot.for_local(
                run_id=run_id, platform=platform_name
            ),
        )

    output_root = WorkflowOutputRoot.from_workflow_run(
        run_id=run_id, platform=platform_name, github_repository=github_repository
    )
    if transport == "http":
        return HTTPBackend(output_root=output_root)
    return S3Backend(output_root=output_root)


def create_backend_from_env(
    run_id: Optional[str] = None,
    github_repository: Optional[str] = None,
    platform: Optional[str] = None,
) -> ArtifactBackend:
    """Create the writable backend for the current run.

    Equivalent to ``create_backend(..., transport="auto")``: LocalDirectoryBackend
    when THEROCK_LOCAL_STAGING_DIR is set, S3Backend otherwise. Callers that need
    a specific transport (notably the read-only HTTP one) should call
    ``create_backend`` directly.
    """
    return create_backend(
        run_id=run_id,
        github_repository=github_repository,
        platform=platform,
        transport="auto",
    )
