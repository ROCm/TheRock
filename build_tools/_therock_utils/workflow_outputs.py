"""CI workflow outputs layout specification.

This module defines the canonical directory structure for all outputs from a
GitHub Actions workflow run. All tools that read or write workflow outputs
should use this module to compute paths.

See docs/development/workflow_outputs.md for the full layout reference.

Note: this file is NOT bundled into the Lambda deployment package (only
storage_backend.py and storage_location.py are). Top-level imports here
do not need to be Lambda-safe, but keep optional dependencies deferred
(imported inside the function that needs them) so this module remains
importable in environments where those packages are not installed.

A "workflow output" is anything produced by a CI workflow run:
- Build artifacts (.tar.xz, .tar.zst archives)
- Logs (.log files, ninja_logs.tar.gz)
- Manifests (therock_manifest.json)
- Python packages (.whl, .tar.gz)
- Reports (build_observability.html, test reports)

Usage::

    from _therock_utils.workflow_outputs import WorkflowOutputRoot

    # Inside a CI workflow (env vars provide bucket info, no API call)
    root = WorkflowOutputRoot.from_workflow_run(run_id="12345", platform="linux")

    # Fetching artifacts from another run (API call for fork/cutover detection)
    root = WorkflowOutputRoot.from_workflow_run(
        run_id="12345", platform="linux", lookup_workflow_run=True
    )

    # For local development/testing
    root = WorkflowOutputRoot.for_local(run_id="local", platform="linux")

    # Get locations for various outputs
    loc = root.artifact("blas_lib_gfx94X.tar.xz")
    print(loc.s3_uri)       # s3://therock-ci-artifacts/12345-linux/blas_lib_gfx94X.tar.xz
    print(loc.https_url)    # https://therock-ci-artifacts.s3.amazonaws.com/...
    print(loc.local_path(Path("/tmp/staging")))  # /tmp/staging/12345-linux/...
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
import platform as platform_module

# Add build_tools to path for sibling package imports.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.storage_location import StorageLocation
from _therock_utils.s3_buckets import get_artifacts_bucket_config_for_workflow_run


# ---------------------------------------------------------------------------
# WorkflowOutputRoot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkflowOutputRoot:
    """Root location for all outputs from a single CI workflow run.

    This is the single source of truth for computing paths to workflow outputs.
    Each method returns a `StorageLocation` that can be resolved to
    S3 URIs, HTTPS URLs, or local paths as needed.

    The class is immutable (frozen) to ensure path computation is deterministic.
    """

    bucket: str
    """S3 bucket name (e.g., 'therock-ci-artifacts')."""

    external_repo: str
    """Repository prefix (e.g., '' for ROCm/TheRock, 'owner-repo/' for forks)."""

    run_id: str
    """GitHub Actions workflow run ID (e.g., '12345678901')."""

    platform: str
    """Platform name ('linux' or 'windows')."""

    # -- Root -------------------------------------------------------------------

    @property
    def prefix(self) -> str:
        """Relative path prefix for this run (no trailing slash).

        This is the common root for all outputs from this run.
        """
        return f"{self.external_repo}{self.run_id}-{self.platform}"

    def root(self) -> StorageLocation:
        """Location for the run output root (where build artifacts live)."""
        return StorageLocation(self.bucket, self.prefix)

    # -- Build artifacts --------------------------------------------------------

    def artifact(self, filename: str) -> StorageLocation:
        """Location for a build artifact file.

        Args:
            filename: Artifact filename (e.g., 'blas_lib_gfx94X.tar.xz')
        """
        return StorageLocation(self.bucket, f"{self.prefix}/{filename}")

    def artifact_index(self) -> StorageLocation:
        """Location for the artifact index HTML (server-side generated)."""
        return StorageLocation(self.bucket, f"{self.prefix}/index.html")

    # -- Logs -------------------------------------------------------------------
    #
    # The log directory contains all build logs, reports, and profiling data
    # for an artifact group. log_dir() gives the directory root; the
    # remaining methods address well-known files within that subtree.

    def log_dir(self, artifact_group: str) -> StorageLocation:
        """Location for a log directory.

        The directory typically contains build.log, ninja_logs.tar.gz,
        build_observability.html (when generated), index.html, and a
        therock-build-prof/ subdirectory with resource profiling data.

        Args:
            artifact_group: Build variant (e.g., 'gfx94X-dcgpu')
        """
        return StorageLocation(self.bucket, f"{self.prefix}/logs/{artifact_group}")

    def log_stage_dir(
        self, stage_name: str, amdgpu_family: str = ""
    ) -> StorageLocation:
        """Location for a multi-arch stage log directory.

        Multi-arch CI uploads logs per stage (and optionally per GPU family)
        rather than per artifact_group. Generic stages get a single directory;
        per-arch stages (e.g., math-libs) get a subdirectory per family.

        Args:
            stage_name: Build stage (e.g., 'foundation', 'math-libs')
            amdgpu_family: GPU family (e.g., 'gfx1151'). Empty for generic stages.
        """
        if amdgpu_family:
            return StorageLocation(
                self.bucket, f"{self.prefix}/logs/{stage_name}/{amdgpu_family}"
            )
        return StorageLocation(self.bucket, f"{self.prefix}/logs/{stage_name}")

    def log_file(self, artifact_group: str, filename: str) -> StorageLocation:
        """Location for a specific file within the log_dir() subtree.

        Args:
            artifact_group: Build variant (e.g., 'gfx94X-dcgpu')
            filename: Log filename (e.g., 'build.log', 'ninja_logs.tar.gz')
        """
        return StorageLocation(
            self.bucket, f"{self.prefix}/logs/{artifact_group}/{filename}"
        )

    def log_index(self, artifact_group: str) -> StorageLocation:
        """Location for the log directory index HTML (within log_dir())."""
        return StorageLocation(
            self.bucket, f"{self.prefix}/logs/{artifact_group}/index.html"
        )

    def log_root_index(self) -> StorageLocation:
        """Location for the root log index HTML (server-side generated)."""
        return StorageLocation(self.bucket, f"{self.prefix}/logs/index.html")

    def build_observability(self, artifact_group: str) -> StorageLocation:
        """Location for build observability HTML (within log_dir())."""
        return StorageLocation(
            self.bucket,
            f"{self.prefix}/logs/{artifact_group}/build_observability.html",
        )

    # -- Manifests --------------------------------------------------------------

    def manifest_dir(self, artifact_group: str) -> StorageLocation:
        """Location for the manifests directory for an artifact group.

        Args:
            artifact_group: Build variant (e.g., 'gfx94X-dcgpu')
        """
        return StorageLocation(self.bucket, f"{self.prefix}/manifests/{artifact_group}")

    def manifest(self, artifact_group: str) -> StorageLocation:
        """Location for therock_manifest.json.

        Args:
            artifact_group: Build variant (e.g., 'gfx94X-dcgpu')
        """
        return StorageLocation(
            self.bucket,
            f"{self.prefix}/manifests/{artifact_group}/therock_manifest.json",
        )

    # -- Python packages --------------------------------------------------------

    def python_packages(self, artifact_group: str = "") -> StorageLocation:
        """Location for the Python packages directory.

        Args:
            artifact_group: Build variant (e.g., 'gfx110X-all'). If empty,
                packages are stored directly under python/ (used for
                multi-arch builds where run_id already uniquely identifies
                the build).
        """
        suffix = f"/{artifact_group}" if artifact_group else ""
        return StorageLocation(self.bucket, f"{self.prefix}/python{suffix}")

    # -- Tarballs ---------------------------------------------------------------

    def tarballs(self) -> StorageLocation:
        """Location for the tarballs directory."""
        return StorageLocation(self.bucket, f"{self.prefix}/tarballs")

    # -- Factories --------------------------------------------------------------

    @classmethod
    def from_workflow_run(
        cls,
        run_id: str,
        platform: str,
        github_repository: str | None = None,
        workflow_run: dict | None = None,
        lookup_workflow_run: bool = False,
    ) -> "WorkflowOutputRoot":
        """Create from CI workflow context.

        Determines the S3 bucket and external_repo prefix from repository
        metadata and environment variables.

        Args:
            run_id: GitHub Actions workflow run ID.
            platform: Platform name ('linux' or 'windows').
            github_repository: Repository in 'owner/repo' format. If None,
                reads GITHUB_REPOSITORY env var (default: 'ROCm/TheRock').
            workflow_run: Optional workflow run dict from GitHub API. If
                provided, uses it directly for fork detection and bucket
                cutover dating (no API call).
            lookup_workflow_run: If True and ``workflow_run`` is not provided,
                fetches the workflow run from the GitHub API using ``run_id``.
                Most callers running inside their own CI workflow do not need
                this — environment variables suffice. Set this when looking up
                another repository's workflow run (e.g. fetching artifacts).
        """
        workflow_run_id = (
            run_id if lookup_workflow_run and workflow_run is None else None
        )
        external_repo, bucket = _retrieve_bucket_info(
            github_repository=github_repository,
            workflow_run_id=workflow_run_id,
            workflow_run=workflow_run,
        )
        return cls(
            bucket=bucket,
            external_repo=external_repo,
            run_id=run_id,
            platform=platform,
        )

    @classmethod
    def for_local(
        cls,
        run_id: str = "local",
        platform: str | None = None,
        bucket: str = "local",
    ) -> "WorkflowOutputRoot":
        """Create for local development/testing.

        Args:
            run_id: Run identifier (default: 'local').
            platform: Platform name. If None, detects from current system.
            bucket: Bucket name placeholder (default: 'local').
        """
        if platform is None:
            platform = platform_module.system().lower()
        return cls(
            bucket=bucket,
            external_repo="",
            run_id=run_id,
            platform=platform,
        )


def _retrieve_bucket_info(
    github_repository: str | None = None,
    workflow_run_id: str | None = None,
    workflow_run: dict | None = None,
) -> tuple[str, str]:
    """Determine S3 bucket and external_repo prefix for a workflow run.

    This is an internal implementation detail — use
    `WorkflowOutputRoot.from_workflow_run` instead.

    Returns:
        Tuple of ``(external_repo, bucket)`` where:
        - external_repo: ``''`` for ROCm/TheRock, or ``'{owner}-{repo}/'``
        - bucket: S3 bucket name
    """
    if not github_repository:
        github_repository = os.environ.get("GITHUB_REPOSITORY", "ROCm/TheRock")

    artifact_bucket_config = get_artifacts_bucket_config_for_workflow_run(
        github_repository=github_repository,
        workflow_run_id=workflow_run_id,
        workflow_run=workflow_run,
    )
    owner, repo_name = github_repository.split("/")
    external_repo = (
        f"{owner}-{repo_name}/"
        if artifact_bucket_config.name == "therock-ci-artifacts-external"
        else ""
    )
    return (external_repo, artifact_bucket_config.name)
