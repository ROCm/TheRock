# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Inventory of S3 buckets used by CI/CD systems and related functions.

See docs/development/s3_buckets.md.
"""

from dataclasses import dataclass, field, replace
import os
import sys


def _log(*args, **kwargs):
    """Log to stdout with flush for CI visibility."""
    print(*args, **kwargs)
    sys.stdout.flush()


@dataclass(frozen=True)
class CdnRule:
    """Maps an S3 key prefix to the public CDN URL prefix that fronts it.

    A bucket may have several rules when different key prefixes are fronted by
    different CDN paths (e.g. ``v4/deb/`` and ``v4/rpm/``). Resolution picks the
    longest matching ``key_prefix``, so a bucket-wide rule (``key_prefix=""``)
    can coexist with more specific ones.
    """

    key_prefix: str
    """S3 key prefix this rule applies to (e.g. 'v4/whl/'). Empty matches the whole bucket."""

    url_prefix: str
    """Public URL replacing key_prefix (e.g. 'https://rocm.nightlies.amd.com/whl-multi-arch/')."""

    def __post_init__(self):
        key_prefix = self.key_prefix
        url_prefix = self.url_prefix
        # Validate rather than coerce: these values can come from a downstream
        # registry file (see THEROCK_S3_BUCKETS_FILE) and a malformed entry
        # would otherwise put a broken link in a CI job summary.
        if key_prefix.startswith("/"):
            raise ValueError(
                f"CdnRule key_prefix must not start with '/' (S3 keys have no "
                f"leading slash), got {key_prefix!r}"
            )
        if not url_prefix.startswith("https://"):
            raise ValueError(
                f"CdnRule url_prefix must start with 'https://', got {url_prefix!r}"
            )
        # Normalize to trailing slashes so prefix swapping never joins or splits
        # a path component.
        if key_prefix and not key_prefix.endswith("/"):
            key_prefix += "/"
        if not url_prefix.endswith("/"):
            url_prefix += "/"
        object.__setattr__(self, "key_prefix", key_prefix)
        object.__setattr__(self, "url_prefix", url_prefix)


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

    key_prefix: str = field(default="")
    """Key prefix prepended to every object this bucket stores (e.g. 'v3/artifacts/').

    Normalized to a trailing '/' when non-empty. Empty for all of TheRock's own
    buckets, which store at the bucket root. Downstream repositories that share a
    bucket layout under a versioned prefix set this via a registry file.
    Independent of ``cdn_rules``: TheRock's release buckets have no ``key_prefix``
    but do have prefix-specific CDN rules.
    """

    cdn_rules: tuple[CdnRule, ...] = field(default=())
    """CDN mappings for this bucket, if any. A tuple so the dataclass stays hashable."""

    namespace_external_repos: bool = field(default=False)
    """Whether uploads are namespaced by '{owner}-{repo}/' (shared external bucket)."""

    def __post_init__(self):
        # Same contract as CdnRule.key_prefix, enforced in the same place: the
        # trailing slash is what keeps prefix concatenation from joining two path
        # components (key_prefix='v3' would yield 'v34242-linux'). Enforced on the
        # dataclass rather than only where configs are parsed, so a config built
        # directly in Python cannot violate an invariant the docstrings promise.
        key_prefix = self.key_prefix
        if key_prefix.startswith("/"):
            raise ValueError(
                f"S3BucketConfig key_prefix must not start with '/' (S3 keys have "
                f"no leading slash), got {key_prefix!r}"
            )
        if key_prefix and not key_prefix.endswith("/"):
            object.__setattr__(self, "key_prefix", key_prefix + "/")

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


# CloudFront distribution roots fronting the release buckets. These same URLs are
# already user-facing elsewhere in the tree (e.g. setup_venv.py's package index map).
_DEV_CDN = "https://rocm.devreleases.amd.com/"
_NIGHTLY_CDN = "https://rocm.nightlies.amd.com/"
_PRERELEASE_CDN = "https://rocm.prereleases.amd.com/"
_RELEASE_CDN = "https://repo.amd.com/rocm/"


# The key prefixes below come from publish_rocm_to_release_buckets.py, which writes
# python packages to "v4/whl" and tarballs to "v4/tarball" for every release type,
# and native packages to "v4/{deb,rpm}/{date}-{run_id}" for dev and nightly.
#
# Deliberately NOT mapped, because the CDN layout is not a prefix rewrite of the
# bucket and a wrong rule would silently produce a link to the wrong file:
#   * prerelease/release native packages - those CDNs serve a distro-partitioned
#     apt/dnf repo (debian12/, ubuntu2204/, rhel8/, gpg/, ...), not v4/packages/{deb,rpm}/.
#   * ASAN prefixes (v4/tarball-asan/, v4/packages-asan/) - no CDN is published.
#   * The artifacts buckets - no CDN, matching the '-' column in s3_buckets.md.
# A bucket with no matching rule falls back to its raw S3 URL, which is the
# behavior every caller had before CDN rules existed.
#
# All 12 rules below were checked against the live CloudFront distributions on
# 2026-08-03; see the pull request for what was and was not proven.
def _whl_cdn_rules(cdn: str) -> tuple[CdnRule, ...]:
    return (CdnRule("v4/whl/", cdn + "whl-multi-arch/"),)


def _tarball_cdn_rules(cdn: str) -> tuple[CdnRule, ...]:
    return (CdnRule("v4/tarball/", cdn + "tarball-multi-arch/"),)


def _package_cdn_rules(cdn: str) -> tuple[CdnRule, ...]:
    return (
        CdnRule("v4/deb/", cdn + "packages-multi-arch/deb/"),
        CdnRule("v4/rpm/", cdn + "packages-multi-arch/rpm/"),
    )


s3_bucket_configs = [
    # CI (external repos use OIDC with therock-ci-external; fork PRs use runner base credentials)
    S3BucketConfig("therock-ci-artifacts", iam_role="therock-ci"),
    S3BucketConfig(
        "therock-ci-artifacts-external",
        iam_role="therock-ci-external",
        namespace_external_repos=True,
    ),
    # Release type "dev"
    S3BucketConfig("therock-dev-artifacts", iam_role="therock-dev"),
    S3BucketConfig(
        "therock-dev-packages",
        iam_role="therock-dev",
        cdn_rules=_package_cdn_rules(_DEV_CDN),
    ),
    S3BucketConfig(
        "therock-dev-python",
        iam_role="therock-dev",
        cdn_rules=_whl_cdn_rules(_DEV_CDN),
    ),
    S3BucketConfig(
        "therock-dev-tarball",
        iam_role="therock-dev",
        cdn_rules=_tarball_cdn_rules(_DEV_CDN),
    ),
    # Release type "nightly"
    S3BucketConfig("therock-nightly-artifacts", iam_role="therock-nightly"),
    S3BucketConfig(
        "therock-nightly-packages",
        iam_role="therock-nightly",
        cdn_rules=_package_cdn_rules(_NIGHTLY_CDN),
    ),
    S3BucketConfig(
        "therock-nightly-python",
        iam_role="therock-nightly",
        cdn_rules=_whl_cdn_rules(_NIGHTLY_CDN),
    ),
    S3BucketConfig(
        "therock-nightly-tarball",
        iam_role="therock-nightly",
        cdn_rules=_tarball_cdn_rules(_NIGHTLY_CDN),
    ),
    # Release type "prerelease"
    S3BucketConfig("therock-prerelease-artifacts", iam_role="therock-prerelease"),
    # TODO: therock-prerelease-packages has a CDN, but it serves a distro-partitioned
    # repo rather than a rewrite of this bucket's v4/packages/{deb,rpm}/ layout.
    S3BucketConfig("therock-prerelease-packages", iam_role="therock-prerelease"),
    S3BucketConfig(
        "therock-prerelease-python",
        iam_role="therock-prerelease",
        cdn_rules=_whl_cdn_rules(_PRERELEASE_CDN),
    ),
    S3BucketConfig(
        "therock-prerelease-tarball",
        iam_role="therock-prerelease",
        cdn_rules=_tarball_cdn_rules(_PRERELEASE_CDN),
    ),
    # Release type "release" (no automated credentials for uploading)
    S3BucketConfig("therock-release-artifacts", iam_role=None),
    # TODO: see the therock-prerelease-packages note above.
    S3BucketConfig("therock-release-packages", iam_role=None),
    S3BucketConfig(
        "therock-release-python",
        iam_role=None,
        cdn_rules=_whl_cdn_rules(_RELEASE_CDN),
    ),
    S3BucketConfig(
        "therock-release-tarball",
        iam_role=None,
        cdn_rules=_tarball_cdn_rules(_RELEASE_CDN),
    ),
]


_ALLOWED_ARTIFACT_RELEASE_TYPES = {"ci", "dev", "nightly", "prerelease"}

_ALLOWED_RELEASE_TYPES = {"dev", "nightly", "prerelease"}

_ALLOWED_RELEASE_BUCKET_TYPES = {"tarball", "python", "packages"}


_bucket_registry_cache: dict[str, S3BucketConfig] | None = None


def _bucket_registry() -> dict[str, S3BucketConfig]:
    """Bucket configs by name, built on first use.

    Built lazily rather than at import time so that a malformed downstream
    registry file raises where a caller can report it, instead of breaking the
    import of every module that reaches for a bucket config at module scope.
    """
    global _bucket_registry_cache
    if _bucket_registry_cache is None:
        _bucket_registry_cache = {c.name: c for c in s3_bucket_configs}
    return _bucket_registry_cache


def reset_bucket_registry() -> None:
    """Discard the cached registry so it is rebuilt on next use (for tests)."""
    global _bucket_registry_cache
    _bucket_registry_cache = None


def lookup_bucket_config(name: str) -> S3BucketConfig | None:
    """Look up a bucket config by name, or None if the bucket is unknown."""
    return _bucket_registry().get(name)


def require_bucket_config(name: str) -> S3BucketConfig:
    """Look up a bucket config by name.

    Raises:
        KeyError: If no bucket with that name is registered.
    """
    config = lookup_bucket_config(name)
    if config is None:
        known = ", ".join(sorted(_bucket_registry()))
        raise KeyError(
            f"Unknown S3 bucket {name!r}. Known buckets: {known}. "
            f"Buckets outside TheRock can be registered via a JSON registry file."
        )
    return config


def all_bucket_configs() -> tuple[S3BucketConfig, ...]:
    """Every registered bucket config, in registration order."""
    return tuple(_bucket_registry().values())


def resolve_public_url(bucket: str, relative_path: str, *, default: str) -> str:
    """Resolve the public (CDN) URL for an object, falling back to ``default``.

    The longest matching ``CdnRule.key_prefix`` wins, so a bucket-wide rule can
    coexist with prefix-specific ones. Buckets that are unknown or have no
    matching rule resolve to ``default`` (normally the raw S3 URL).

    Args:
        bucket: S3 bucket name.
        relative_path: Full S3 key, including any bucket ``key_prefix``.
        default: URL to return when no CDN rule applies.
    """
    config = lookup_bucket_config(bucket)
    if config is None:
        return default
    for rule in sorted(config.cdn_rules, key=lambda r: len(r.key_prefix), reverse=True):
        if relative_path.startswith(rule.key_prefix):
            return rule.url_prefix + relative_path[len(rule.key_prefix) :]
    return default


def get_artifacts_bucket_config(
    release_type: str,
    repository: str,
    is_pr_from_fork: bool,
) -> S3BucketConfig:
    """Look up the artifacts bucket config for a repository.

    Args:
        release_type: "ci", "dev", "nightly", or "prerelease".
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
    else:
        bucket_name = f"therock-{release_type}-artifacts"
    return require_bucket_config(bucket_name)


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
    return require_bucket_config(bucket_name)


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
        # Use dataclasses.replace rather than rebuilding field-by-field, so that
        # fields added to S3BucketConfig later are carried over instead of being
        # silently reset to their defaults on every fork PR.
        config = replace(config, iam_role=None)

    return config
