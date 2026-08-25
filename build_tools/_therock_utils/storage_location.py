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
    loc.public_url    # CDN URL if the bucket has one, else https_url
    loc.download_url  # what a credential-less tool should fetch: raw S3 for a
                      # publicly readable bucket, the CDN otherwise
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
    - ``.https_url`` - Raw S3 URL (``https://bucket.s3.amazonaws.com/...``)
    - ``.public_url`` - CDN URL where one is configured, else ``.https_url``
    - ``.download_url`` - For pip/apt/curl: raw S3 when the bucket allows
      anonymous reads, the CDN when it does not
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
        """Raw S3 HTTPS URL, bypassing any CDN.

        Prefer ``.public_url`` for links a human will click. This stays the raw
        S3 URL for machine-consumed URLs (package indexes, apt/dnf base URLs) and
        for existence probes, where a CDN cache could give a stale answer and
        where CI reads deliberately avoid CloudFront data-transfer charges. See
        docs/development/s3_buckets.md.
        """
        return f"https://{self.bucket}.s3.amazonaws.com/{self.relative_path}"

    @property
    def public_url(self) -> str:
        """Public URL for browser access, via the bucket's CDN when it has one.

        Falls back to ``.https_url`` for buckets with no CDN rule covering this
        key, which is every bucket TheRock's CI writes build outputs to.
        """
        # Deferred import: storage_location is bundled into the artifact index
        # Lambda and must stay free of import-time project dependencies.
        from _therock_utils.s3_buckets import resolve_public_url

        return resolve_public_url(
            self.bucket, self.relative_path, default=self.https_url
        )

    @property
    def download_url(self) -> str:
        """URL a tool without AWS credentials should fetch this object from.

        For pip indexes, apt/dnf base URLs and plain downloads - the URLs that
        end up in a command line rather than in a job summary.

        Prefers the raw S3 URL, because CI reads there avoid CloudFront
        data-transfer charges, and falls back to the CDN for buckets that refuse
        anonymous S3 reads. That choice is driven by the bucket's
        ``anonymous_s3_read`` flag rather than by a rule about the call site: the
        prerelease and release buckets are private (ROCm/TheRock#2139) and so are
        downstream repositories' buckets, and handing any of those a raw S3 URL
        produces a link that answers 403.

        Unlike ``.public_url`` this raises rather than silently returning a URL
        that cannot be fetched: a private bucket with no CDN rule covering the
        key has no public download URL at all, and discovering that from a 403
        in a build log is worse than discovering it here.
        """
        # Deferred import, for the same reason as public_url above.
        from _therock_utils.s3_buckets import cdn_url_for, lookup_bucket_config

        config = lookup_bucket_config(self.bucket)
        if config is None or config.anonymous_s3_read:
            return self.https_url

        url = cdn_url_for(self.bucket, self.relative_path)
        if url is None:
            raise ValueError(
                f"Bucket {self.bucket!r} refuses anonymous S3 reads and has no CDN "
                f"rule covering {self.relative_path!r}, so this object has no "
                f"public download URL. Add a cdn_rule for it in "
                f"_therock_utils/s3_buckets.py, or read it with credentials."
            )
        return url

    def local_path(self, staging_dir: Path) -> Path:
        """Local filesystem path for this location.

        Args:
            staging_dir: Base directory for local staging.

        Returns:
            Full path: ``{staging_dir}/{relative_path}``
        """
        return staging_dir / self.relative_path
