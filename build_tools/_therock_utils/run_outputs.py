"""TheRock CI run outputs layout specification.

This module defines the canonical directory structure for all outputs from a
GitHub Actions workflow run. All tools that read or write run outputs should
use this module to compute paths.

See the documentation at /docs/development/run_outputs_layout.md.

A "run output" is anything produced by a CI workflow run:
- Build artifacts (.tar.xz, .tar.zst archives)
- Logs (.log files, ninja_logs.tar.gz)
- Manifests (therock_manifest.json)
- Reports (build_time_analysis.html, test reports)

Layout Structure
----------------
There is a 1:1 mapping between a GitHub Actions workflow run ID and a run
outputs directory. The structure is:

    {root}/
    ├── {name}_{component}_{family}.tar.xz        # Build artifacts (at root)
    ├── {name}_{component}_{family}.tar.xz.sha256sum
    ├── index-{artifact_group}.html               # Per-group artifact index
    ├── logs/{artifact_group}/
    │   ├── *.log                                 # Build logs
    │   ├── ninja_logs.tar.gz                     # Ninja timing logs
    │   ├── index.html                            # Log index
    │   └── build_time_analysis.html              # Build timing (Linux)
    └── manifests/{artifact_group}/
        └── therock_manifest.json                 # Build manifest

Where:
- {root} = {external_repo}{run_id}-{platform}
- {external_repo} = "" for ROCm/TheRock, or "{owner}-{repo}/" for forks/external
- {run_id} = GitHub Actions workflow run ID
- {platform} = "linux" or "windows"
- {name} = artifact name (e.g., "blas", "core-hip", "amd-llvm")
- {component} = dev|lib|run|test|dbg|doc
- {family} = "generic" or GPU family (gfx94X, gfx1100, etc.)
- {artifact_group} = build variant (e.g., "gfx94X-dcgpu", "gfx950-dcgpu-asan")

Multi-Platform / Multi-Group Organization
-----------------------------------------
A single workflow run may produce outputs for multiple platforms and artifact
groups. Each platform gets its own root directory:

    s3://therock-ci-artifacts/
    ├── 12345678901-linux/
    │   ├── *.tar.xz (artifacts for all Linux artifact groups)
    │   ├── index-gfx94X-dcgpu.html
    │   ├── index-gfx110X-all.html
    │   ├── logs/gfx94X-dcgpu/
    │   ├── logs/gfx110X-all/
    │   └── ...
    └── 12345678901-windows/
        ├── *.tar.xz (artifacts for all Windows artifact groups)
        ├── index-gfx110X-dgpu.html
        └── ...

Multiple CI jobs (different GPU families, build variants) upload to the same
run directory, differentiated by artifact_group in subdirectory names and
index filenames.

Usage
-----
    from _therock_utils.run_outputs import RunOutputRoot

    # From CI environment
    root = RunOutputRoot.from_workflow_run(
        run_id="12345678901",
        platform="linux",
    )

    # Get paths for various outputs
    print(root.s3_uri)                              # s3://bucket/12345678901-linux
    print(root.artifact_index_url("gfx94X-dcgpu"))  # https://...index-gfx94X-dcgpu.html
    print(root.logs_s3_uri("gfx94X-dcgpu"))         # s3://bucket/.../logs/gfx94X-dcgpu

    # For local development/testing
    root = RunOutputRoot.for_local(run_id="local", platform="linux")
    local_dir = root.local_path(Path("/tmp/staging"))
"""

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import platform as platform_module

# Add build_tools to path for sibling package imports
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from github_actions.github_actions_utils import gha_query_workflow_run_by_id


def _log(*args, **kwargs):
    """Log to stdout with flush for CI visibility."""
    print(*args, **kwargs)
    sys.stdout.flush()


# Cutover date for bucket naming change (TheRock #2046).
# Workflows before this date used therock-artifacts, after use therock-ci-artifacts.
_BUCKET_CUTOVER_DATE = datetime.fromisoformat("2025-11-11T16:18:48+00:00")


def _retrieve_bucket_info(
    github_repository: str | None = None,
    workflow_run_id: str | None = None,
    workflow_run: dict | None = None,
) -> tuple[str, str]:
    """Determine S3 bucket and external_repo prefix for a workflow run.

    This is an internal implementation detail - use RunOutputRoot.from_workflow_run()
    instead of calling this directly.

    Bucket selection is based on:
    - Repository (ROCm/TheRock vs external)
    - Whether the PR is from a fork
    - RELEASE_TYPE environment variable (nightly, release)
    - Workflow run date (for legacy bucket compatibility)

    Args:
        github_repository: Repository in 'owner/repo' format. If None,
            uses GITHUB_REPOSITORY env var or defaults to 'ROCm/TheRock'.
        workflow_run_id: Workflow run ID. If provided without workflow_run,
            fetches workflow data from GitHub API.
        workflow_run: Workflow run dict from GitHub API. If provided,
            avoids an API call.

    Returns:
        Tuple of (external_repo, bucket) where:
        - external_repo: '' for ROCm/TheRock, or '{owner}-{repo}/' for forks/external
        - bucket: S3 bucket name
    """
    _log("Retrieving bucket info...")

    if github_repository:
        _log(f"  (explicit) github_repository: {github_repository}")
    else:
        github_repository = os.getenv("GITHUB_REPOSITORY", "ROCm/TheRock")
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
        is_pr_from_fork = os.getenv("IS_PR_FROM_FORK", "false") == "true"
        _log(f"  (implicit) is_pr_from_fork  : {is_pr_from_fork}")

    owner, repo_name = github_repository.split("/")
    external_repo = (
        ""
        if repo_name == "TheRock" and owner == "ROCm" and not is_pr_from_fork
        else f"{owner}-{repo_name}/"
    )

    release_type = os.getenv("RELEASE_TYPE")
    if release_type:
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


@dataclass(frozen=True)
class RunOutputRoot:
    """Root location for a workflow run's outputs.

    This is the single source of truth for computing paths to all outputs
    from a CI workflow run. Use this class instead of manually constructing
    S3 keys or URLs.

    The class is immutable (frozen) to ensure path computation is deterministic.
    """

    bucket: str
    """S3 bucket name (e.g., 'therock-ci-artifacts')."""

    external_repo: str
    """Repository prefix for namespacing (e.g., 'ROCm-TheRock/' or '')."""

    run_id: str
    """GitHub Actions workflow run ID (e.g., '12345678901')."""

    platform: str
    """Platform name ('linux' or 'windows')."""

    # -------------------------------------------------------------------------
    # Root paths
    # -------------------------------------------------------------------------

    @property
    def prefix(self) -> str:
        """S3 key prefix for this run (no leading/trailing slash).

        This is the relative path from the bucket root to the run directory.
        """
        return f"{self.external_repo}{self.run_id}-{self.platform}"

    @property
    def s3_uri(self) -> str:
        """S3 URI for the run root (e.g., 's3://bucket/12345-linux')."""
        return f"s3://{self.bucket}/{self.prefix}"

    @property
    def https_url(self) -> str:
        """Public HTTPS URL for the run root."""
        return f"https://{self.bucket}.s3.amazonaws.com/{self.prefix}"

    def local_path(self, staging_dir: Path) -> Path:
        """Local directory path mirroring S3 structure.

        Args:
            staging_dir: Base directory for local staging.

        Returns:
            Path like {staging_dir}/{external_repo}{run_id}-{platform}/
        """
        return staging_dir / self.prefix

    # -------------------------------------------------------------------------
    # Build artifacts (.tar.xz, .tar.zst)
    #
    # Artifacts are stored at the root of the run directory.
    # This is the current layout; if we migrate to artifacts/ subdirectory,
    # only these methods need to change.
    # -------------------------------------------------------------------------

    def artifact_s3_key(self, filename: str) -> str:
        """S3 key for a build artifact file.

        Args:
            filename: Artifact filename (e.g., 'blas_lib_gfx94X.tar.xz')
        """
        return f"{self.prefix}/{filename}"

    def artifact_s3_uri(self, filename: str) -> str:
        """S3 URI for a build artifact file."""
        return f"{self.s3_uri}/{filename}"

    def artifact_https_url(self, filename: str) -> str:
        """Public HTTPS URL for a build artifact file."""
        return f"{self.https_url}/{filename}"

    def artifact_index_s3_key(self, artifact_group: str) -> str:
        """S3 key for the artifact index HTML file."""
        return f"{self.prefix}/index-{artifact_group}.html"

    def artifact_index_s3_uri(self, artifact_group: str) -> str:
        """S3 URI for the artifact index HTML file (for uploads)."""
        return f"s3://{self.bucket}/{self.artifact_index_s3_key(artifact_group)}"

    def artifact_index_url(self, artifact_group: str) -> str:
        """Public URL for the artifact index HTML file."""
        return f"{self.https_url}/index-{artifact_group}.html"

    # -------------------------------------------------------------------------
    # Logs
    # -------------------------------------------------------------------------

    def logs_prefix(self, artifact_group: str) -> str:
        """S3 key prefix for logs directory (no trailing slash)."""
        return f"{self.prefix}/logs/{artifact_group}"

    def logs_s3_uri(self, artifact_group: str) -> str:
        """S3 URI for logs directory."""
        return f"s3://{self.bucket}/{self.logs_prefix(artifact_group)}"

    def logs_https_url(self, artifact_group: str) -> str:
        """Public HTTPS URL for logs directory."""
        return f"{self.https_url}/logs/{artifact_group}"

    def log_file_s3_key(self, artifact_group: str, filename: str) -> str:
        """S3 key for a specific log file."""
        return f"{self.logs_prefix(artifact_group)}/{filename}"

    def log_index_url(self, artifact_group: str) -> str:
        """Public URL for the log index HTML file."""
        return f"{self.logs_https_url(artifact_group)}/index.html"

    def build_time_analysis_url(self, artifact_group: str) -> str:
        """Public URL for build time analysis HTML (Linux only)."""
        return f"{self.logs_https_url(artifact_group)}/build_time_analysis.html"

    # -------------------------------------------------------------------------
    # Manifests
    # -------------------------------------------------------------------------

    def manifests_prefix(self, artifact_group: str) -> str:
        """S3 key prefix for manifests directory (no trailing slash)."""
        return f"{self.prefix}/manifests/{artifact_group}"

    def manifest_s3_key(self, artifact_group: str) -> str:
        """S3 key for therock_manifest.json."""
        return f"{self.manifests_prefix(artifact_group)}/therock_manifest.json"

    def manifest_s3_uri(self, artifact_group: str) -> str:
        """S3 URI for therock_manifest.json."""
        return f"s3://{self.bucket}/{self.manifest_s3_key(artifact_group)}"

    def manifest_url(self, artifact_group: str) -> str:
        """Public URL for therock_manifest.json."""
        return f"{self.https_url}/manifests/{artifact_group}/therock_manifest.json"

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_workflow_run(
        cls,
        run_id: str,
        platform: str,
        github_repository: str | None = None,
        workflow_run: dict | None = None,
    ) -> "RunOutputRoot":
        """Create from workflow context.

        Determines the appropriate S3 bucket and external_repo prefix based
        on the repository and workflow run metadata.

        Args:
            run_id: GitHub Actions workflow run ID.
            platform: Platform name ('linux' or 'windows').
            github_repository: Repository in 'owner/repo' format. If None,
                uses GITHUB_REPOSITORY env var or defaults to 'ROCm/TheRock'.
            workflow_run: Optional workflow run dict from GitHub API. If
                provided, avoids a GitHub API call.

        Returns:
            RunOutputRoot configured for the workflow run.
        """
        external_repo, bucket = _retrieve_bucket_info(
            github_repository=github_repository,
            workflow_run_id=run_id,
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
    ) -> "RunOutputRoot":
        """Create for local development/testing.

        Creates a RunOutputRoot with placeholder values suitable for local
        staging directories. The bucket name is set to 'local' by default.

        Args:
            run_id: Run identifier (default: 'local').
            platform: Platform name. If None, detects from current system.
            bucket: Bucket name placeholder (default: 'local').

        Returns:
            RunOutputRoot configured for local use.
        """
        if platform is None:
            platform = platform_module.system().lower()
        return cls(
            bucket=bucket,
            external_repo="",
            run_id=run_id,
            platform=platform,
        )
