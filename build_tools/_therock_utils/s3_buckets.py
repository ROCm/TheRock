# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Inventory of S3 buckets used by CI/CD systems and related functions.

See docs/development/s3_buckets.md.
"""

from dataclasses import dataclass, field
import os
import sys


def _log(*args, **kwargs):
    """Log to stdout with flush for CI visibility."""
    print(*args, **kwargs)
    sys.stdout.flush()


@dataclass(frozen=True)
class S3BucketConfig:
    """Metadata for a single bucket in S3"""

    name: str
    """S3 bucket name (e.g. 'therock-ci-artifacts')"""

    region: str = field(default="us-east-2")
    """Region in S3 (e.g. 'us-east-2')"""

    iam_account: str | None = field(default="692859939525")
    """IAM account for write_access_iam_role"""

    iam_role: str | None = field(default=None)
    """IAM role name that grants write access to this bucket (e.g. 'therock-ci'), if any"""

    @property
    def write_access_iam_role(self) -> str | None:
        """IAM role granting write access to the bucket"""
        if not self.iam_role:
            return None
        if not self.iam_account:
            raise ValueError(
                f"Bucket {self.name!r} has iam_role={self.iam_role!r} but no iam_account"
            )
        return f"arn:aws:iam::{self.iam_account}:role/{self.iam_role}"


s3_bucket_configs = [
    # CI (self-hosted runners include credentials for therock-ci-artifacts-external)
    S3BucketConfig("therock-ci-artifacts", iam_role="therock-ci"),
    S3BucketConfig("therock-ci-artifacts-external", iam_role=None),
    # Release type "dev"
    S3BucketConfig("therock-dev-artifacts", iam_role="therock-dev"),
    S3BucketConfig("therock-dev-packages", iam_role="therock-dev"),
    S3BucketConfig("therock-dev-python", iam_role="therock-dev"),
    S3BucketConfig("therock-dev-tarball", iam_role="therock-dev"),
    # Release type "nightly"
    S3BucketConfig("therock-nightly-artifacts", iam_role="therock-nightly"),
    S3BucketConfig("therock-nightly-packages", iam_role="therock-nightly"),
    S3BucketConfig("therock-nightly-python", iam_role="therock-nightly"),
    S3BucketConfig("therock-nightly-tarball", iam_role="therock-nightly"),
    # Release type "prerelease"
    S3BucketConfig("therock-prerelease-artifacts", iam_role="therock-prerelease"),
    S3BucketConfig("therock-prerelease-packages", iam_role="therock-prerelease"),
    S3BucketConfig("therock-prerelease-python", iam_role="therock-prerelease"),
    S3BucketConfig("therock-prerelease-tarball", iam_role="therock-prerelease"),
    # Release type "release" (no automated credentials for uploading)
    S3BucketConfig("therock-release-artifacts", iam_role=None),
    S3BucketConfig("therock-release-packages", iam_role=None),
    S3BucketConfig("therock-release-python", iam_role=None),
    S3BucketConfig("therock-release-tarball", iam_role=None),
]


_BUCKET_CONFIGS_BY_NAME = {c.name: c for c in s3_bucket_configs}

_ALLOWED_RELEASE_TYPES = {"dev", "nightly", "prerelease"}

_ALLOWED_RELEASE_BUCKET_TYPES = {"tarball", "python", "packages"}


def get_artifacts_bucket_config(
    release_type: str,
    repository: str,
    is_pr_from_fork: bool,
) -> S3BucketConfig:
    """Look up the artifacts bucket config for a repository.

    Args:
        release_type: "" for CI builds, or "dev", "nightly", "prerelease".
        repository: GitHub repository (e.g. "ROCm/TheRock").
        is_pr_from_fork: Whether this is a PR from a fork.

    Raises:
        ValueError: If release_type is invalid.
    """
    if release_type:
        if release_type not in _ALLOWED_RELEASE_TYPES:
            raise ValueError(
                f"release_type={release_type!r} is invalid, "
                f"expected empty string or one of {_ALLOWED_RELEASE_TYPES}"
            )
        bucket_name = f"therock-{release_type}-artifacts"
    else:
        if is_pr_from_fork or repository != "ROCm/TheRock":
            bucket_name = "therock-ci-artifacts-external"
        else:
            bucket_name = "therock-ci-artifacts"
    return _BUCKET_CONFIGS_BY_NAME[bucket_name]


def get_release_bucket_config(
    release_type: str,
    bucket_type: str,
) -> S3BucketConfig:
    """Look up the release bucket config for a given release type and bucket type.

    Args:
        release_type: "dev", "nightly", or "prerelease".
        bucket_type: "tarball", "python", or "packages".

    Returns:
        S3BucketConfig for the bucket ``therock-{release_type}-{bucket_type}``.

    Raises:
        ValueError: If release_type or bucket_type is invalid.
    """
    if release_type not in _ALLOWED_RELEASE_TYPES:
        raise ValueError(
            f"release_type={release_type!r} is invalid, "
            f"expected one of {_ALLOWED_RELEASE_TYPES}"
        )
    if bucket_type not in _ALLOWED_RELEASE_BUCKET_TYPES:
        raise ValueError(
            f"bucket_type={bucket_type!r} is invalid, "
            f"expected one of {_ALLOWED_RELEASE_BUCKET_TYPES}"
        )
    bucket_name = f"therock-{release_type}-{bucket_type}"
    return _BUCKET_CONFIGS_BY_NAME[bucket_name]


def _is_current_run_pr_from_fork() -> bool:
    """Check if the current workflow run is a pull request from a fork.

    Reads the GitHub event payload to check the .fork property on the
    head repo, matching the behavior of the GitHub Actions expression
    ``github.event.pull_request.head.repo.fork``.

    Returns False for non-pull_request events or if the event payload
    is not available (e.g. local development).
    """
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    if event_name != "pull_request":
        return False

    if not os.environ.get("GITHUB_EVENT_PATH"):
        return False

    # Deferred import: github_actions is optional in some environments; only
    # needed when resolving fork state from the on-disk event payload.
    from github_actions.github_actions_api import gha_load_github_event

    event = gha_load_github_event()

    return bool(
        event.get("pull_request", {}).get("head", {}).get("repo", {}).get("fork", False)
    )


def get_artifacts_bucket_config_for_workflow_run(
    github_repository: str,
    release_type: str | None = None,
    workflow_run_id: str | None = None,
    workflow_run: dict | None = None,
) -> S3BucketConfig:
    """Look up the artifacts bucket config for a workflow run.

    Combines environment-based inputs (RELEASE_TYPE, event payload) with
    optional workflow run metadata from the GitHub API to determine the
    correct artifacts bucket.

    Args:
        github_repository: GitHub repository (e.g. "ROCm/TheRock").
        release_type: Release type override. If None, reads RELEASE_TYPE
            from the environment (default: empty string = CI build).
        workflow_run_id: If set and ``workflow_run`` is None, fetches the
            workflow run from the GitHub API for fork detection.
        workflow_run: Optional workflow run dict from GitHub API. If
            provided, used directly for fork detection (no API call).
    """
    _log("Retrieving bucket info for workflow run...")
    _log(f"  github_repository: {github_repository}")

    if release_type is None:
        release_type = os.environ.get("RELEASE_TYPE", "")
    if release_type:
        _log(f"  release_type: {release_type}")

    # Fetch workflow_run from API if not provided but workflow_run_id is set.
    # Deferred import: github_actions is an optional dependency not available in
    # all environments (e.g. local dev without the GHA support package installed).
    if workflow_run is None and workflow_run_id is not None:
        from github_actions.github_actions_api import gha_query_workflow_run_by_id

        workflow_run = gha_query_workflow_run_by_id(github_repository, workflow_run_id)

    # Extract metadata from workflow_run if available
    if workflow_run is not None:
        _log(f"  workflow_run_id: {workflow_run['id']}")
        head_github_repository = workflow_run["head_repository"]["full_name"]
        is_pr_from_fork = head_github_repository != github_repository
        _log(f"  head_github_repository: {head_github_repository}")
        _log(f"  is_pr_from_fork: {is_pr_from_fork}")
    else:
        is_pr_from_fork = _is_current_run_pr_from_fork()
        _log(f"  is_pr_from_fork: {is_pr_from_fork}")

    config = get_artifacts_bucket_config(
        release_type=release_type,
        repository=github_repository,
        is_pr_from_fork=is_pr_from_fork,
    )
    _log(f"  bucket: {config.name}")
    return config
