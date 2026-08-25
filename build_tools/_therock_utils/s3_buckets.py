# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Inventory of S3 buckets used by CI/CD systems and related functions.

See docs/development/s3_buckets.md.

Repositories outside TheRock that reuse these build tools against their own
buckets can extend the in-tree inventory with a JSON registry file, pointed at
by the ``THEROCK_S3_BUCKETS_FILE`` environment variable or by an explicit
``--bucket-config-file`` argument. See ``load_bucket_registry_file`` for the
schema and merge rules.
"""

from dataclasses import dataclass, field, replace
import json
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

    anonymous_s3_read: bool = field(default=True)
    """Whether the bucket can be read over raw S3 without credentials.

    Machine-consumed URLs (a pip index, an apt/dnf base URL, a tarball download)
    prefer the raw S3 URL when this is True, because CI reads there avoid
    CloudFront data-transfer charges - see docs/development/s3_buckets.md. When
    it is False the CDN is the only public way in, so those URLs must go through
    ``cdn_rules`` instead; falling back to raw S3 would hand out a URL that
    answers 403.

    This is a fact about the bucket rather than a policy about the caller, which
    is why it lives here. The prerelease buckets grant no anonymous read
    (ROCm/TheRock#2139), and downstream repositories whose buckets are entirely
    private set it via a registry file.
    """

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


# ---------------------------------------------------------------------------
# repo.amd.com streams and products (RFC0012)
#
# Every stream is served from its own subdomain, and the folder hierarchy under
# each one is identical. Both facts come straight from
# docs/rfcs/RFC0012-Repo-Structure.md, so the URL is derived from a formula
# rather than transcribed per stream: a stream that has not finished cutting
# over yet needs no follow-up edit here when it does.
# ---------------------------------------------------------------------------
_ALLOWED_RELEASE_PRODUCTS = {"core", "pytorch", "jax"}

_RELEASE_STREAM_BY_TYPE = {
    "dev": "dev",
    "dev-bkc": "bkc",
    "nightly": "nightly",
    "nightly-bkc": "bkc",
    "prerelease": "rc",
}

# The key prefix every product bucket stores under, stripped from the public URL:
# s3://therock-repo-amd-nightly-core/v5/rocm/core/tarball/X is served at
# https://nightly.repo.amd.com/rocm/core/tarball/X.
_PRODUCT_KEY_PREFIX = "v5/"


def release_stream_url(stream: str) -> str:
    """Public root for a repo.amd.com stream, e.g. 'https://nightly.repo.amd.com/'."""
    return f"https://{stream}.repo.amd.com/"


def _product_cdn_rules(stream: str) -> tuple[CdnRule, ...]:
    return (CdnRule(_PRODUCT_KEY_PREFIX, release_stream_url(stream)),)


def _product_release_bucket_configs() -> list[S3BucketConfig]:
    """Final publication buckets, one per (stream, product).

    Cross-account (324352301041) and reachable only through the stream CDN;
    anonymous S3 reads are refused, verified 2026-08-25.
    """
    return [
        S3BucketConfig(
            f"therock-repo-amd-{stream}-{product}",
            iam_account="324352301041",
            iam_role=f"therock-repo-{stream}-{product}",
            key_prefix=_PRODUCT_KEY_PREFIX,
            cdn_rules=_product_cdn_rules(stream),
            anonymous_s3_read=False,
        )
        # Sorted for a stable inventory order; the set is small and fixed.
        for stream in sorted(set(_RELEASE_STREAM_BY_TYPE.values()))
        for product in sorted(_ALLOWED_RELEASE_PRODUCTS)
    ]


# ---------------------------------------------------------------------------
# Legacy release CDNs (pre-RFC0012 layout)
#
# These distributions front the therock-{release_type}-{python,tarball,packages}
# buckets. Those buckets are no longer publication targets - since the
# repo.amd.com rewire, publish_rocm_to_release_buckets.py writes to the product
# buckets below - but they still serve every release made under the old layout,
# so the mappings stay. See docs/development/s3_buckets.md.
# ---------------------------------------------------------------------------
_DEV_CDN = "https://rocm.devreleases.amd.com/"
_NIGHTLY_CDN = "https://rocm.nightlies.amd.com/"
_PRERELEASE_CDN = "https://rocm.prereleases.amd.com/"
_RELEASE_CDN = "https://repo.amd.com/rocm/"


# The key prefixes below come from publish_rocm_to_release_buckets.py, which wrote
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
        anonymous_s3_read=False,
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
        anonymous_s3_read=False,
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
    S3BucketConfig(
        "therock-prerelease-packages",
        iam_role="therock-prerelease",
        anonymous_s3_read=False,
    ),
    S3BucketConfig(
        "therock-prerelease-python",
        iam_role="therock-prerelease",
        cdn_rules=_whl_cdn_rules(_PRERELEASE_CDN),
        anonymous_s3_read=False,
    ),
    S3BucketConfig(
        "therock-prerelease-tarball",
        iam_role="therock-prerelease",
        cdn_rules=_tarball_cdn_rules(_PRERELEASE_CDN),
        anonymous_s3_read=False,
    ),
    # Release type "release" (no automated credentials for uploading)
    S3BucketConfig("therock-release-artifacts", iam_role=None),
    # TODO: see the therock-prerelease-packages note above.
    S3BucketConfig("therock-release-packages", iam_role=None, anonymous_s3_read=False),
    S3BucketConfig(
        "therock-release-python",
        iam_role=None,
        cdn_rules=_whl_cdn_rules(_RELEASE_CDN),
        anonymous_s3_read=False,
    ),
    S3BucketConfig(
        "therock-release-tarball",
        iam_role=None,
        cdn_rules=_tarball_cdn_rules(_RELEASE_CDN),
        anonymous_s3_read=False,
    ),
    # Final publication buckets for the repo.amd.com product layout.
    *_product_release_bucket_configs(),
]


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

# _ALLOWED_RELEASE_PRODUCTS and _RELEASE_STREAM_BY_TYPE are defined above the
# bucket inventory, which is generated from them.


# ---------------------------------------------------------------------------
# Bucket registry
#
# The in-tree inventory above, optionally extended by a downstream JSON file.
# ---------------------------------------------------------------------------

BUCKET_REGISTRY_ENV_VAR = "THEROCK_S3_BUCKETS_FILE"

_SUPPORTED_REGISTRY_VERSIONS = (1,)

_REGISTRY_TOP_LEVEL_KEYS = {
    "version",
    "buckets",
    "artifacts_buckets",
    "release_buckets",
    "product_release_buckets",
}
_REGISTRY_BUCKET_KEYS = {
    "name",
    "region",
    "iam_account",
    "iam_role",
    "key_prefix",
    "cdn_rules",
    "namespace_external_repos",
    "anonymous_s3_read",
    "override",
}
_REGISTRY_CDN_RULE_KEYS = {"key_prefix", "url_prefix"}

# Selection slots for get_artifacts_bucket_config. "ci" and "ci-external" are
# separate slots because the shared external bucket is chosen by fork state
# rather than by release type, and a downstream overriding one should have to
# say what it wants for the other rather than inherit it silently.
_ARTIFACTS_SELECTION_SLOTS = _ALLOWED_ARTIFACT_RELEASE_TYPES | {"ci-external"}


class BucketRegistryError(Exception):
    """A bucket registry file is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class _BucketRegistry:
    """Resolved bucket inventory plus any downstream selection overrides."""

    buckets: dict[str, S3BucketConfig]
    artifacts_buckets: dict[str, str]
    release_buckets: dict[str, dict[str, str]]
    product_release_buckets: dict[str, dict[str, str]]


_registry_cache: _BucketRegistry | None = None
_bucket_config_file_override: str | None = None


def set_bucket_config_file(path: str | os.PathLike | None) -> None:
    """Point the registry at an explicit file, overriding the environment variable.

    Backs the ``--bucket-config-file`` argument. Like ``THEROCK_S3_BUCKETS_FILE``
    this sets process-wide state - it writes a module global that every later
    lookup reads - but the value is stated by the invocation rather than
    inherited from the environment, so it cannot be picked up unnoticed from a
    parent process. Prefer it wherever a single invocation should be
    self-describing; the environment variable exists for wrapper scripts that
    shell out to many entry points and cannot pass a flag to each.

    Passing None clears the override and falls back to the environment variable.
    """
    global _bucket_config_file_override
    _bucket_config_file_override = os.fspath(path) if path is not None else None
    reset_bucket_registry()


def _registry_file_path() -> str | None:
    """The registry file to load, if any. The explicit override wins."""
    env_value = os.environ.get(BUCKET_REGISTRY_ENV_VAR) or None
    if _bucket_config_file_override is not None:
        if env_value and env_value != _bucket_config_file_override:
            _log(
                f"[INFO] Bucket registry: using --bucket-config-file "
                f"{_bucket_config_file_override!r}, ignoring "
                f"{BUCKET_REGISTRY_ENV_VAR}={env_value!r}"
            )
        return _bucket_config_file_override
    return env_value


def _reject_unknown_keys(obj: dict, allowed: set[str], what: str, path: str) -> None:
    """Fail on keys we do not recognize.

    A typo'd ``cdn_rule`` would otherwise mean "this bucket has no CDN", which is
    exactly the silently-wrong-URL failure this whole mechanism exists to prevent.
    """
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise BucketRegistryError(
            f"{path}: unknown {what} key(s): {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )


def _parse_cdn_rules(entries, bucket_name: str, path: str) -> tuple[CdnRule, ...]:
    if not isinstance(entries, list):
        raise BucketRegistryError(
            f"{path}: bucket {bucket_name!r} 'cdn_rules' must be a list"
        )
    rules = []
    for index, entry in enumerate(entries):
        where = f"bucket {bucket_name!r} cdn_rules[{index}]"
        if not isinstance(entry, dict):
            raise BucketRegistryError(f"{path}: {where} must be an object")
        _reject_unknown_keys(entry, _REGISTRY_CDN_RULE_KEYS, where, path)
        for key in sorted(_REGISTRY_CDN_RULE_KEYS):
            if key not in entry:
                raise BucketRegistryError(f"{path}: {where} is missing {key!r}")
        try:
            rules.append(CdnRule(entry["key_prefix"], entry["url_prefix"]))
        except (ValueError, AttributeError) as e:
            raise BucketRegistryError(f"{path}: {where}: {e}") from e
    return tuple(rules)


def _parse_bucket(entry, path: str) -> tuple[S3BucketConfig, bool]:
    """Parse one bucket entry. Returns (config, override_requested)."""
    if not isinstance(entry, dict):
        raise BucketRegistryError(f"{path}: each entry in 'buckets' must be an object")
    name = entry.get("name")
    if not isinstance(name, str) or not name:
        raise BucketRegistryError(
            f"{path}: each entry in 'buckets' needs a non-empty string 'name'"
        )
    _reject_unknown_keys(entry, _REGISTRY_BUCKET_KEYS, f"bucket {name!r}", path)

    key_prefix = entry.get("key_prefix", "")
    if not isinstance(key_prefix, str):
        raise BucketRegistryError(
            f"{path}: bucket {name!r} 'key_prefix' must be a string"
        )

    namespace_external_repos = entry.get("namespace_external_repos", False)
    if not isinstance(namespace_external_repos, bool):
        raise BucketRegistryError(
            f"{path}: bucket {name!r} 'namespace_external_repos' must be a boolean"
        )

    anonymous_s3_read = entry.get("anonymous_s3_read", True)
    if not isinstance(anonymous_s3_read, bool):
        raise BucketRegistryError(
            f"{path}: bucket {name!r} 'anonymous_s3_read' must be a boolean"
        )

    # S3BucketConfig.__post_init__ validates and normalizes key_prefix; wrap its
    # ValueError with the file path, the same way _parse_cdn_rules does for CdnRule.
    try:
        config = S3BucketConfig(
            name=name,
            region=entry.get("region", "us-east-2"),
            iam_account=entry.get("iam_account", "692859939525"),
            iam_role=entry.get("iam_role"),
            key_prefix=key_prefix,
            cdn_rules=_parse_cdn_rules(entry.get("cdn_rules", []), name, path),
            namespace_external_repos=namespace_external_repos,
            anonymous_s3_read=anonymous_s3_read,
        )
    except ValueError as e:
        raise BucketRegistryError(f"{path}: bucket {name!r}: {e}") from e
    return config, bool(entry.get("override", False))


def _parse_selection_map(
    raw,
    allowed_keys: set[str],
    buckets: dict[str, S3BucketConfig],
    what: str,
    path: str,
) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise BucketRegistryError(f"{path}: {what!r} must be an object")
    _reject_unknown_keys(raw, allowed_keys, what, path)
    result = {}
    for key, value in raw.items():
        if not isinstance(value, str):
            raise BucketRegistryError(
                f"{path}: {what}[{key!r}] must be a bucket name string"
            )
        if value not in buckets:
            raise BucketRegistryError(
                f"{path}: {what}[{key!r}] names unregistered bucket {value!r}. "
                f"Add it to 'buckets' in the same file."
            )
        result[key] = value
    return result


def load_bucket_registry_file(path: str) -> _BucketRegistry:
    """Load a registry file and merge it over the in-tree inventory.

    Schema::

        {
          "version": 1,
          "buckets": [
            {
              "name": "therock-npi-artifacts",
              "key_prefix": "v3/artifacts/",
              "iam_role": "therock-npi",
              "cdn_rules": [
                {"key_prefix": "v3/artifacts/",
                 "url_prefix": "https://genesis.example.com/artifacts/"}
              ]
            }
          ],
          "artifacts_buckets": {
            "ci": "therock-npi-artifacts",
            "ci-external": "therock-npi-artifacts"
          },
          "release_buckets": {"nightly": {"python": "therock-npi-python"}}
        }

    ``buckets`` registers bucket metadata. ``artifacts_buckets`` and
    ``release_buckets`` override *selection*: which bucket the lookup functions
    choose, which registration alone cannot express because those functions
    compute a name from a formula.

    Note the two CI slots. ``get_artifacts_bucket_config`` picks ``ci-external``
    for fork PRs *and* for any repository other than ROCm/TheRock, so a
    downstream repo lands there for all of its own CI. Set both slots (to the
    same bucket, if fork PRs need no separate destination). They are separate
    keys on purpose: overriding only ``ci`` must not silently redirect
    untrusted fork uploads into a trusted bucket.

    Merge rules:

    * Additive. A name already in the in-tree inventory is rejected unless the
      entry sets ``"override": true``, in which case it fully replaces the
      in-tree entry (no field-wise merge) and the replacement is logged.
      Silently retargeting a production bucket from an inherited environment
      variable is the worst failure this mechanism could have, so it is opt-in
      and loud.
    * Unknown keys at any level are an error, not ignored.
    * Selection overrides must name a bucket registered in the same file or
      in-tree.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise BucketRegistryError(f"Bucket registry file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise BucketRegistryError(f"{path}: invalid JSON: {e}") from e
    except OSError as e:
        raise BucketRegistryError(f"{path}: could not be read: {e}") from e

    if not isinstance(data, dict):
        raise BucketRegistryError(f"{path}: top level must be a JSON object")
    _reject_unknown_keys(data, _REGISTRY_TOP_LEVEL_KEYS, "top-level", path)

    version = data.get("version")
    if version not in _SUPPORTED_REGISTRY_VERSIONS:
        raise BucketRegistryError(
            f"{path}: unsupported 'version' {version!r}; "
            f"this build supports {list(_SUPPORTED_REGISTRY_VERSIONS)}"
        )

    buckets = {c.name: c for c in s3_bucket_configs}
    in_tree_names = set(buckets)
    seen_in_file: set[str] = set()

    raw_buckets = data.get("buckets", [])
    if not isinstance(raw_buckets, list):
        raise BucketRegistryError(f"{path}: 'buckets' must be a list")
    for entry in raw_buckets:
        config, override_requested = _parse_bucket(entry, path)
        if config.name in seen_in_file:
            raise BucketRegistryError(
                f"{path}: bucket {config.name!r} is defined more than once"
            )
        seen_in_file.add(config.name)
        if config.name in in_tree_names:
            if not override_requested:
                raise BucketRegistryError(
                    f"{path}: bucket {config.name!r} already exists in TheRock's "
                    f'inventory. Set "override": true on this entry to replace it.'
                )
            _log(
                f"[WARNING] Bucket registry: {path} replaces TheRock's built-in "
                f"config for bucket {config.name!r}"
            )
        buckets[config.name] = config

    artifacts_buckets = _parse_selection_map(
        data.get("artifacts_buckets", {}),
        _ARTIFACTS_SELECTION_SLOTS,
        buckets,
        "artifacts_buckets",
        path,
    )

    raw_release = data.get("release_buckets", {})
    if not isinstance(raw_release, dict):
        raise BucketRegistryError(f"{path}: 'release_buckets' must be an object")
    _reject_unknown_keys(raw_release, _ALLOWED_RELEASE_TYPES, "release_buckets", path)
    release_buckets = {
        release_type: _parse_selection_map(
            inner,
            _ALLOWED_RELEASE_BUCKET_TYPES,
            buckets,
            f"release_buckets[{release_type!r}]",
            path,
        )
        for release_type, inner in raw_release.items()
    }

    raw_products = data.get("product_release_buckets", {})
    if not isinstance(raw_products, dict):
        raise BucketRegistryError(
            f"{path}: 'product_release_buckets' must be an object"
        )
    _reject_unknown_keys(
        raw_products, _ALLOWED_RELEASE_TYPES, "product_release_buckets", path
    )
    product_release_buckets = {
        release_type: _parse_selection_map(
            inner,
            _ALLOWED_RELEASE_PRODUCTS,
            buckets,
            f"product_release_buckets[{release_type!r}]",
            path,
        )
        for release_type, inner in raw_products.items()
    }

    _log(f"[INFO] Bucket registry: loaded {len(seen_in_file)} bucket(s) from {path}")
    return _BucketRegistry(
        buckets, artifacts_buckets, release_buckets, product_release_buckets
    )


def _registry() -> _BucketRegistry:
    """The resolved registry, built on first use.

    Built lazily rather than at import time so that a malformed downstream
    registry file raises where a caller can report it, instead of breaking the
    import of every module that reaches for a bucket config at module scope.
    """
    global _registry_cache
    if _registry_cache is None:
        path = _registry_file_path()
        if path:
            _registry_cache = load_bucket_registry_file(path)
        else:
            _registry_cache = _BucketRegistry(
                {c.name: c for c in s3_bucket_configs}, {}, {}, {}
            )
    return _registry_cache


def _bucket_registry() -> dict[str, S3BucketConfig]:
    """Bucket configs by name."""
    return _registry().buckets


def reset_bucket_registry() -> None:
    """Discard the cached registry so it is rebuilt on next use (for tests)."""
    global _registry_cache
    _registry_cache = None


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
            f"Buckets outside TheRock can be registered via a JSON registry file "
            f"named by {BUCKET_REGISTRY_ENV_VAR} or --bucket-config-file."
        )
    return config


def all_bucket_configs() -> tuple[S3BucketConfig, ...]:
    """Every registered bucket config, in registration order."""
    return tuple(_bucket_registry().values())


def cdn_url_for(bucket: str, relative_path: str) -> str | None:
    """CDN URL for an object, or None when no rule covers it.

    The longest matching ``CdnRule.key_prefix`` wins, so a bucket-wide rule can
    coexist with prefix-specific ones.

    This is the ``str | None`` form, for callers that must not fall back to a raw
    S3 URL: a value handed to pip, or an object in a bucket that refuses
    anonymous reads. Callers that do have a sensible fallback want
    ``resolve_public_url``.

    Args:
        bucket: S3 bucket name.
        relative_path: Full S3 key, including any bucket ``key_prefix``.
    """
    config = lookup_bucket_config(bucket)
    if config is None:
        return None
    for rule in sorted(config.cdn_rules, key=lambda r: len(r.key_prefix), reverse=True):
        if relative_path.startswith(rule.key_prefix):
            return rule.url_prefix + relative_path[len(rule.key_prefix) :]
    return None


def resolve_public_url(bucket: str, relative_path: str, *, default: str) -> str:
    """Resolve the public (CDN) URL for an object, falling back to ``default``.

    Buckets that are unknown or have no matching rule resolve to ``default``
    (normally the raw S3 URL).

    Args:
        bucket: S3 bucket name.
        relative_path: Full S3 key, including any bucket ``key_prefix``.
        default: URL to return when no CDN rule applies.
    """
    url = cdn_url_for(bucket, relative_path)
    return default if url is None else url


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

    if release_type == "ci" and (is_pr_from_fork or repository != "ROCm/TheRock"):
        slot = "ci-external"
        bucket_name = "therock-ci-artifacts-external"
    else:
        # BKC builds share the dev/nightly artifact buckets, but keep their own
        # selection slot so a downstream registry can redirect a BKC channel
        # without also redirecting the channel it shares a bucket with.
        slot = release_type
        if release_type == "dev-bkc":
            bucket_name = "therock-dev-artifacts"
        elif release_type == "nightly-bkc":
            bucket_name = "therock-nightly-artifacts"
        else:
            bucket_name = f"therock-{release_type}-artifacts"

    # A downstream registry file can override which bucket a slot selects;
    # registering a bucket by name is not enough, because the name is computed
    # here from a formula the downstream repo does not follow.
    bucket_name = _registry().artifacts_buckets.get(slot, bucket_name)
    return require_bucket_config(bucket_name)


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
        the corresponding dev or nightly bucket. A registry file can redirect
        the slot via ``release_buckets``.

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
    # See the note in get_artifacts_bucket_config: selection, not registration.
    # Keyed by the release_type as given, so a BKC channel can be redirected
    # independently of the dev/nightly bucket it otherwise shares.
    bucket_name = (
        _registry().release_buckets.get(release_type, {}).get(bucket_type, bucket_name)
    )
    return require_bucket_config(bucket_name)


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
    # Looked up rather than constructed here, so these buckets carry the same
    # key_prefix and cdn_rules as every other entry and StorageLocation can
    # derive their public URLs. The config is otherwise identical to the one
    # this function used to build inline.
    bucket_name = f"therock-repo-amd-{stream}-{product}"
    # See the note in get_artifacts_bucket_config: selection, not registration.
    bucket_name = (
        _registry()
        .product_release_buckets.get(release_type, {})
        .get(product, bucket_name)
    )
    return require_bucket_config(bucket_name)


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
        # Use dataclasses.replace rather than rebuilding field-by-field, so that
        # fields added to S3BucketConfig later are carried over instead of being
        # silently reset to their defaults on every fork PR.
        config = replace(config, iam_role=None)

    return config
