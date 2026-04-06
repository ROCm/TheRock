"""Backend-agnostic storage location.

A ``StorageLocation`` represents a file or directory in S3 (or a local staging
directory) without coupling to any particular layout or upload/download
direction.  It is the bridge between path computation modules (like
``workflow_outputs.WorkflowOutputRoot``) and I/O modules (``storage_backend``,
``artifact_backend``).

Usage::

    from _therock_utils.storage_location import StorageLocation

    loc = StorageLocation("my-bucket", "some/path/file.tar.xz")
    loc.s3_uri        # "s3://my-bucket/some/path/file.tar.xz"
    loc.https_url     # "https://my-bucket.s3.amazonaws.com/some/path/file.tar.xz"
    loc.local_path(Path("/tmp/staging"))  # Path("/tmp/staging/some/path/file.tar.xz")
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StorageLocation:
    """A location that can be resolved to S3 URI, HTTPS URL, or local path.

    Represents a single file or directory in a backend-agnostic way.
    Use the properties/methods to get the representation you need:

    - ``.s3_uri`` - For AWS CLI uploads (``s3://bucket/path/file.tar.xz``)
    - ``.https_url`` - For public links (``https://bucket.s3.amazonaws.com/...``)
    - ``.local_path(staging_dir)`` - For local testing (``Path("/tmp/staging/...")``)
    - ``.relative_path`` - Backend-agnostic relative path from the bucket/staging root
    """

    bucket: str
    """S3 bucket name (used for S3 URI and HTTPS URL construction)."""

    relative_path: str
    """Relative path from bucket/staging root (e.g., '12345-linux/file.tar.xz')."""

    @property
    def s3_uri(self) -> str:
        """S3 URI (e.g., ``s3://bucket/path/file``)."""
        return f"s3://{self.bucket}/{self.relative_path}"

    @property
    def https_url(self) -> str:
        """Public HTTPS URL for browser access."""
        return f"https://{self.bucket}.s3.amazonaws.com/{self.relative_path}"

    def local_path(self, staging_dir: Path) -> Path:
        """Local filesystem path for this location.

        Args:
            staging_dir: Base directory for local staging.

        Returns:
            Full path: ``{staging_dir}/{relative_path}``
        """
        return staging_dir / self.relative_path

    def cdn_url(
        self,
        base_url: str,
        strip_prefix: str = "",
        cdn_prefix: str = "",
    ) -> str:
        """CDN URL, optionally remapping the storage path prefix to a CDN prefix.

        Strips ``strip_prefix`` from the start of ``relative_path`` (if present),
        then prepends ``cdn_prefix``, and finally prepends ``base_url``.

        This lets private workflows surface artifacts through a CDN whose path
        layout differs from the storage key layout. For example, files stored under
        ``v3/artifacts/{run_id}-linux/...`` may be served as
        ``artifacts/{run_id}-linux/...`` on the CDN.

        Args:
            base_url: CDN base URL without trailing slash
                (e.g. ``'https://artifacts.example.com'``).
            strip_prefix: Path prefix to strip from ``relative_path``
                (e.g. ``'v3/artifacts'``).  No-op if empty or not matched.
            cdn_prefix: CDN path prefix to prepend after stripping
                (e.g. ``'artifacts'``).  No-op if empty.

        Returns:
            Full CDN URL (e.g.
            ``'https://artifacts.example.com/artifacts/12345-linux/file.tar.xz'``).
        """
        path = self.relative_path
        if strip_prefix and path.startswith(f"{strip_prefix}/"):
            path = path[len(strip_prefix) + 1 :]
        if cdn_prefix:
            path = f"{cdn_prefix}/{path}"
        return f"{base_url}/{path}"
