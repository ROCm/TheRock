"""CI workflow outputs layout specification.

This module defines the canonical directory structure for all outputs from a
GitHub Actions workflow run. All tools that read or write workflow outputs
should use this module to compute paths.

See docs/development/workflow_outputs.md for the full layout reference.

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
from datetime import datetime, timezone
from pathlib import Path
import platform as platform_module

# Add build_tools to path for sibling package imports.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.storage_location import StorageLocation
from github_actions.github_actions_utils import gha_query_workflow_run_by_id


def _log(*args, **kwargs):
    """Log to stdout with flush for CI visibility."""
    print(*args, **kwargs)
    sys.stdout.flush()


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

    def artifact_index(self, artifact_group: str) -> StorageLocation:
        """Location for the per-group artifact index HTML.

        Args:
            artifact_group: Build variant (e.g., 'gfx94X-dcgpu')
        """
        return StorageLocation(
            self.bucket, f"{self.prefix}/index-{artifact_group}.html"
        )

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

    def python_packages(self, artifact_group: str) -> StorageLocation:
        """Location for the Python packages directory.

        Args:
            artifact_group: Build variant (e.g., 'gfx110X-all')
        """
        return StorageLocation(self.bucket, f"{self.prefix}/python/{artifact_group}")

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


# ---------------------------------------------------------------------------
# Bucket selection logic
# ---------------------------------------------------------------------------

# Cutover date for bucket naming change (TheRock #2046).
# Workflows before this date used therock-artifacts; after, therock-ci-artifacts.
_BUCKET_CUTOVER_DATE = datetime.fromisoformat("2025-11-11T16:18:48+00:00")


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
    _log("Retrieving bucket info...")

    if github_repository:
        _log(f"  (explicit) github_repository: {github_repository}")
    else:
        github_repository = os.environ.get("GITHUB_REPOSITORY", "ROCm/TheRock")
        _log(f"  (implicit) github_repository: {github_repository}")

    # Fetch workflow_run from API if not provided but workflow_run_id is set
    if workflow_run is None and workflow_run_id is not None:
        workflow_run = gha_query_workflow_run_by_id(github_repository, workflow_run_id)

    # Extract metadata from workflow_run if available
    curr_commit_dt = None
    if workflow_run is not None:
        _log(f"  workflow_run_id             : {workflow_run['id']}")
        head_github_repository = workflow_run["head_repository"]["full_name"]
        is_pr_from_fork = head_github_repository != github_repository
        _log(f"  head_github_repository      : {head_github_repository}")
        _log(f"  is_pr_from_fork             : {is_pr_from_fork}")

        curr_commit_dt = datetime.strptime(
            workflow_run["updated_at"], "%Y-%m-%dT%H:%M:%SZ"
        )
        curr_commit_dt = curr_commit_dt.replace(tzinfo=timezone.utc)
    else:
        is_pr_from_fork = os.environ.get("IS_PR_FROM_FORK", "false") == "true"
        _log(f"  (implicit) is_pr_from_fork  : {is_pr_from_fork}")

    owner, repo_name = github_repository.split("/")
    external_repo = (
        ""
        if repo_name == "TheRock" and owner == "ROCm" and not is_pr_from_fork
        else f"{owner}-{repo_name}/"
    )

    release_type = os.environ.get("RELEASE_TYPE")
    if release_type:
        _VALID_RELEASE_TYPES = {"dev", "nightly", "prerelease"}
        if release_type not in _VALID_RELEASE_TYPES:
            raise ValueError(
                f"Invalid RELEASE_TYPE={release_type!r}, "
                f"expected one of {sorted(_VALID_RELEASE_TYPES)}"
            )
        _log(f"  (implicit) RELEASE_TYPE: {release_type}")
        bucket = f"therock-{release_type}-artifacts"
    else:
        if external_repo == "":
            bucket = "therock-ci-artifacts"
            if curr_commit_dt and curr_commit_dt <= _BUCKET_CUTOVER_DATE:
                bucket = "therock-artifacts"
        elif (
            repo_name == "therock-releases-internal"
            and owner == "ROCm"
            and not is_pr_from_fork
        ):
            bucket = "therock-artifacts-internal"
        else:
            bucket = "therock-ci-artifacts-external"
            if curr_commit_dt and curr_commit_dt <= _BUCKET_CUTOVER_DATE:
                bucket = "therock-artifacts-external"

    _log("Retrieved bucket info:")
    _log(f"  external_repo: {external_repo}")
    _log(f"  bucket       : {bucket}")
    return (external_repo, bucket)
