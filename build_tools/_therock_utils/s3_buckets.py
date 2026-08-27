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

    cdn_url: str | None = field(default=None)
    """Base URL of the CloudFront distribution in front of this bucket, if any.

    Downloading through the CDN needs no AWS credentials and is the only public
    way to read buckets that do not grant anonymous S3 read access (such as the
    prerelease buckets). Note that the CDN path segment does not always match
    the S3 key prefix: 'tarball-multi-arch/' on the CDN maps to 'v4/tarball/'
    in the bucket, for example. See docs/development/s3_buckets.md.
    """

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
    # CI (external repos use OIDC with therock-ci-external; fork PRs use runner base credentials)
    S3BucketConfig("therock-ci-artifacts", iam_role="therock-ci"),
    S3BucketConfig("therock-ci-artifacts-external", iam_role="therock-ci-external"),
    # Release type "{dev,nightly}-bkc"; else BKC targets product publication buckets
    S3BucketConfig("therock-bkc-artifacts", iam_role="therock-bkc"),
    # Release type "dev"
    S3BucketConfig("therock-dev-artifacts", iam_role="therock-dev"),
    S3BucketConfig(
        "therock-dev-packages",
        iam_role="therock-dev",
        cdn_url="https://rocm.devreleases.amd.com/packages-multi-arch",
    ),
    S3BucketConfig(
        "therock-dev-python",
        iam_role="therock-dev",
        cdn_url="https://rocm.devreleases.amd.com/whl-multi-arch",
    ),
    S3BucketConfig(
        "therock-dev-tarball",
        iam_role="therock-dev",
        cdn_url="https://rocm.devreleases.amd.com/tarball-multi-arch",
    ),
    # Release type "nightly"
    S3BucketConfig("therock-nightly-artifacts", iam_role="therock-nightly"),
    S3BucketConfig(
        "therock-nightly-packages",
        iam_role="therock-nightly",
        cdn_url="https://rocm.nightlies.amd.com/packages-multi-arch",
    ),
    S3BucketConfig(
        "therock-nightly-python",
        iam_role="therock-nightly",
        cdn_url="https://rocm.nightlies.amd.com/whl-multi-arch",
    ),
    S3BucketConfig(
        "therock-nightly-tarball",
        iam_role="therock-nightly",
        cdn_url="https://rocm.nightlies.amd.com/tarball-multi-arch",
    ),
    # Release type "prerelease"
    S3BucketConfig("therock-prerelease-artifacts", iam_role="therock-prerelease"),
    S3BucketConfig(
        "therock-prerelease-packages",
        iam_role="therock-prerelease",
        cdn_url="https://rocm.prereleases.amd.com/packages-multi-arch",
    ),
    S3BucketConfig(
        "therock-prerelease-python",
        iam_role="therock-prerelease",
        cdn_url="https://rocm.prereleases.amd.com/whl-multi-arch",
    ),
    S3BucketConfig(
        "therock-prerelease-tarball",
        iam_role="therock-prerelease",
        cdn_url="https://rocm.prereleases.amd.com/tarball-multi-arch",
    ),
    # Release type "release" (no automated credentials for uploading)
    S3BucketConfig("therock-release-artifacts", iam_role=None),
    S3BucketConfig(
        "therock-release-packages",
        iam_role=None,
        cdn_url="https://repo.amd.com/rocm/packages-multi-arch",
    ),
    S3BucketConfig(
        "therock-release-python",
        iam_role=None,
        cdn_url="https://repo.amd.com/rocm/whl-multi-arch",
    ),
    S3BucketConfig(
        "therock-release-tarball",
        iam_role=None,
        cdn_url="https://repo.amd.com/rocm/tarball-multi-arch",
    ),
]


_BUCKET_CONFIGS_BY_NAME = {c.name: c for c in s3_bucket_configs}

_ALLOWED_ARTIFACT_RELEASE_TYPES = {
    "ci",
    "dev",
    "dev-bkc",
    "nightly",
    "nightly-bkc",
    "prerelease",
}

_ALLOWED_RELEASE_TYPES = {
    "dev",
    "dev-bkc",
    "nightly",
    "nightly-bkc",
    "prerelease",
}

_ALLOWED_RELEASE_BUCKET_TYPES = {"tarball", "python", "packages"}
_ALLOWED_RELEASE_PRODUCTS = {"core", "pytorch", "jax"}
_RELEASE_STREAM_BY_TYPE = {
    "dev": "dev",
    "dev-bkc": "bkc",
    "nightly": "nightly",
    "nightly-bkc": "bkc",
    "prerelease": "rc",
}


def get_artifacts_bucket_config(
    release_type: str,
    repository: str,
    is_pr_from_fork: bool,
) -> S3BucketConfig:
    """Look up the artifacts bucket config for a repository.

    Args:
        release_type: "ci", "dev", "dev-bkc", "nightly", "nightly-bkc", or
            "prerelease".
        repository: GitHub repository (e.g. "ROCm/TheRock").
        is_pr_from_fork: Whether this is a PR from a fork.

    Raises:
        ValueError: If release_type is invalid.
    """
    if release_type not in _ALLOWED_ARTIFACT_RELEASE_TYPES:
        raise ValueError(
            f"release_type={release_type!r} is invalid, "
            f"expected one of {_ALLOWED_ARTIFACT_RELEASE_TYPES}"
        )

    if release_type == "ci":
        if is_pr_from_fork or repository != "ROCm/TheRock":
            bucket_name = "therock-ci-artifacts-external"
        else:
            bucket_name = "therock-ci-artifacts"
    elif release_type in ("dev-bkc", "nightly-bkc"):
        bucket_name = "therock-bkc-artifacts"
    else:
        bucket_name = f"therock-{release_type}-artifacts"
    return _BUCKET_CONFIGS_BY_NAME[bucket_name]


def get_release_bucket_config(
    release_type: str,
    bucket_type: str,
) -> S3BucketConfig:
    """Look up the release bucket config for a given release type and bucket type.

    Args:
        release_type: "dev", "dev-bkc", "nightly", "nightly-bkc", or
            "prerelease".
        bucket_type: "tarball", "python", or "packages".

    Returns:
        S3BucketConfig for the selected release bucket. BKC release types use
        the corresponding dev or nightly bucket.

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
    if release_type == "dev-bkc":
        bucket_name = f"therock-dev-{bucket_type}"
    elif release_type == "nightly-bkc":
        bucket_name = f"therock-nightly-{bucket_type}"
    else:
        bucket_name = f"therock-{release_type}-{bucket_type}"
    return _BUCKET_CONFIGS_BY_NAME[bucket_name]


def get_release_stream(release_type: str) -> str:
    """Return the external repo.amd.com stream for an internal release type.

    Args:
        release_type: "dev", "dev-bkc", "nightly", "nightly-bkc", or
            "prerelease".

    Raises:
        ValueError: If release_type is invalid.
    """
    try:
        return _RELEASE_STREAM_BY_TYPE[release_type]
    except KeyError as e:
        raise ValueError(
            f"release_type={release_type!r} is invalid, "
            f"expected one of {_ALLOWED_RELEASE_TYPES}"
        ) from e


def get_product_release_bucket_config(
    release_type: str,
    product: str,
) -> S3BucketConfig:
    """Look up the final repo.amd.com product publication bucket.

    Artifact buckets and credentials are intentionally separate from this
    product publication path. This resolver targets the cross-account product
    buckets used for final public release outputs.

    Args:
        release_type: "dev", "dev-bkc", "nightly", "nightly-bkc", or
            "prerelease".
        product: "core", "pytorch", or "jax".

    Raises:
        ValueError: If release_type or product is invalid.
    """
    stream = get_release_stream(release_type)
    if product not in _ALLOWED_RELEASE_PRODUCTS:
        raise ValueError(
            f"product={product!r} is invalid, "
            f"expected one of {_ALLOWED_RELEASE_PRODUCTS}"
        )
    return S3BucketConfig(
        name=f"therock-repo-amd-{stream}-{product}",
        iam_account="324352301041",
        iam_role=f"therock-repo-{stream}-{product}",
    )


def get_release_package_index_url(release_type: str) -> str:
    """Return the aggregate pip index URL for a final release stream."""
    stream = get_release_stream(release_type)
    return f"https://{stream}.repo.amd.com/rocm/whl-next/"


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
            from the environment (default: "ci").
        workflow_run_id: If set and ``workflow_run`` is None, fetches the
            workflow run from the GitHub API for fork detection.
        workflow_run: Optional workflow run dict from GitHub API. If
            provided, used directly for fork detection (no API call).
    """
    _log("Retrieving bucket info for workflow run...")
    _log(f"  github_repository: {github_repository}")

    if release_type is None:
        release_type = os.environ.get("RELEASE_TYPE", "ci")
    _log(f"  release_type: {release_type}")

    # Fetch workflow_run from API if not provided but workflow_run_id is set.
    # Deferred import: github_actions is an optional dependency not available in
    # all environments (e.g. local dev without the GHA support package installed).
    if workflow_run is None and workflow_run_id is not None:
        from github_actions.github_actions_api import (
            GitHubAPIError,
            gha_query_workflow_run_by_id,
        )

        try:
            workflow_run = gha_query_workflow_run_by_id(
                github_repository, workflow_run_id
            )
        except GitHubAPIError as e:
            run_url = (
                f"https://github.com/{github_repository}/actions/runs/{workflow_run_id}"
            )
            raise GitHubAPIError(
                f"Failed to query workflow run {workflow_run_id} in repository "
                f"{github_repository}: {run_url}\n"
                f"  {e}\n"
                f"Hint: Did you mean to specify a different repository with "
                f"--run-github-repo?"
            ) from e

    # Extract metadata from workflow_run if available
    if workflow_run is not None:
        _log(f"  workflow_run_id: {workflow_run['id']}")
        head_github_repository = workflow_run["head_repository"]["full_name"]
        is_pr_from_fork = head_github_repository != github_repository
        _log(f"  head_github_repository: {head_github_repository}")
        _log(f"  is_pr_from_fork: {is_pr_from_fork}")
    else:
        # Deferred import: github_actions is optional in some environments;
        # only needed when resolving fork state from the on-disk event payload.
        from github_actions.github_actions_api import is_current_run_pr_from_fork

        is_pr_from_fork = is_current_run_pr_from_fork()
        _log(f"  is_pr_from_fork: {is_pr_from_fork}")

    config = get_artifacts_bucket_config(
        release_type=release_type,
        repository=github_repository,
        is_pr_from_fork=is_pr_from_fork,
    )
    _log(f"  bucket: {config.name}")

    # For fork PRs, skip OIDC and use runner base credentials instead.
    # Fork PRs cannot assume IAM roles via OIDC because they don't have
    # the required trust relationship. Return a config without an IAM role
    # so the configure-aws-credentials step is skipped.
    if is_pr_from_fork and config.iam_role is not None:
        _log("  Fork PR detected, skipping OIDC (using runner base credentials)")
        config = S3BucketConfig(
            name=config.name,
            region=config.region,
            iam_account=config.iam_account,
            iam_role=None,
        )

    return config
