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
