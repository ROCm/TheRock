"""Backend-agnostic storage location and configuration.

A ``StorageLocation`` represents a file or directory in S3 (or a local staging
directory) without coupling to any particular layout or upload/download
direction.  It is the bridge between path computation modules (like
``workflow_outputs.WorkflowOutputRoot``) and I/O modules (``storage_backend``,
``artifact_backend``).

A ``StorageConfig`` consolidates all URL schema configurations into a single,
immutable object with validation and sensible defaults.

Usage::

    from _therock_utils.storage_location import StorageLocation, StorageConfig

    # Using StorageConfig for custom schemas
    config = StorageConfig(https_url_schema="https://cdn.example.com/{path}")

    loc = StorageLocation("my-bucket", "some/path/file.tar.xz", storage_config=config)
    loc.s3_uri        # "s3://my-bucket/some/path/file.tar.xz"
    loc.https_url     # "https://cdn.example.com/some/path/file.tar.xz"
    loc.local_path(Path("/tmp/staging"))  # Path("/tmp/staging/some/path/file.tar.xz")
"""

import json
import string
from dataclasses import dataclass
from pathlib import Path

# Allowed placeholders for each schema type
URL_SCHEMA_PLACEHOLDERS = frozenset({"bucket", "path"})
BUCKET_SCHEMA_PLACEHOLDERS = frozenset({"release_type"})


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _extract_placeholders(schema: str) -> set[str]:
    """Extract placeholder names from a format string.

    Args:
        schema: A Python format string (e.g., "s3://{bucket}/{path}")

    Returns:
        Set of placeholder names (e.g., {"bucket", "path"})
    """
    formatter = string.Formatter()
    return {field for _, field, _, _ in formatter.parse(schema) if field}


def _validate_schema(schema: str | None, allowed: frozenset[str], name: str) -> None:
    """Validate that schema only uses allowed placeholders.

    Args:
        schema: Format string to validate, or None
        allowed: Set of allowed placeholder names
        name: Schema name for error messages

    Raises:
        ValueError: If schema uses placeholders not in allowed set
    """
    if schema is None:
        return
    used = _extract_placeholders(schema)
    invalid = used - allowed
    if invalid:
        raise ValueError(
            f"{name} uses invalid placeholder(s): {sorted(invalid)}. "
            f"Allowed: {sorted(allowed)}"
        )


# ---------------------------------------------------------------------------
# StorageConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StorageConfig:
    """Configuration for storage URL schema templates.

    Schemas use Python format-string placeholders. Available placeholders:

    For s3_url_schema and https_url_schema:
        {bucket} - S3 bucket name
        {path}   - Relative path within the bucket

    For bucket_schema:
        {release_type} - Release type ('dev', 'nightly', 'prerelease')

    Schemas may omit placeholders (e.g., CDN URLs that ignore {bucket}).
    Using a placeholder not in the allowed list raises ValueError.

    Usage::

        # Use defaults
        config = StorageConfig()

        # Custom CDN (omits {bucket})
        config = StorageConfig(
            https_url_schema="https://cdn.example.com/{path}"
        )

        # Invalid - unknown placeholder
        config = StorageConfig(
            https_url_schema="https://{bucket}.{region}.s3.amazonaws.com/{path}"
        )  # Raises ValueError: https_url_schema uses invalid placeholder(s): ['region']

        # Parse from JSON
        config = StorageConfig.from_json('{"https_url_schema": "https://cdn.example.com/{path}"}')
    """

    s3_url_schema: str = "s3://{bucket}/{path}"
    https_url_schema: str = "https://{bucket}.s3.amazonaws.com/{path}"
    bucket_schema: str = "therock-{release_type}-artifacts"

    def __post_init__(self):
        """Validate schemas at construction time."""
        _validate_schema(self.s3_url_schema, URL_SCHEMA_PLACEHOLDERS, "s3_url_schema")
        _validate_schema(
            self.https_url_schema, URL_SCHEMA_PLACEHOLDERS, "https_url_schema"
        )
        _validate_schema(
            self.bucket_schema, BUCKET_SCHEMA_PLACEHOLDERS, "bucket_schema"
        )

    @classmethod
    def from_json(cls, json_str: str) -> "StorageConfig":
        """Parse StorageConfig from JSON string.

        Args:
            json_str: JSON object with optional keys:
                - s3_url_schema
                - https_url_schema
                - bucket_schema

        Returns:
            StorageConfig instance

        Raises:
            ValueError: If JSON is invalid or contains unknown keys
            json.JSONDecodeError: If JSON syntax is invalid
        """
        data = json.loads(json_str)
        if not isinstance(data, dict):
            raise ValueError("Storage config must be a JSON object")
        allowed_keys = {"s3_url_schema", "https_url_schema", "bucket_schema"}
        unknown = set(data.keys()) - allowed_keys
        if unknown:
            raise ValueError(f"Unknown keys in storage config: {sorted(unknown)}")
        return cls(**data)


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

    storage_config: StorageConfig = StorageConfig()
    """Storage configuration for URL schemas."""

    @property
    def s3_uri(self) -> str:
        """S3 URI (e.g., ``s3://bucket/path/file``)."""
        return self.storage_config.s3_url_schema.format(
            bucket=self.bucket, path=self.relative_path
        )

    @property
    def https_url(self) -> str:
        """Public HTTPS URL for browser access."""
        return self.storage_config.https_url_schema.format(
            bucket=self.bucket, path=self.relative_path
        )

    def local_path(self, staging_dir: Path) -> Path:
        """Local filesystem path for this location.

        Args:
            staging_dir: Base directory for local staging.

        Returns:
            Full path: ``{staging_dir}/{relative_path}``
        """
        return staging_dir / self.relative_path
