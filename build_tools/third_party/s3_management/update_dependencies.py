# Copyright Facebook, Inc. and its affiliates.
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# Forked from https://github.com/pytorch/test-infra/blob/1ffc7f7b3b421b57c380de469e11744f54399f09/s3_management/update_dependencies.py.
# Changes incorporated from https://github.com/pytorch/test-infra/blob/a87d94b148bbd2c68e69e542350099a971f4c8d3/s3_management/update_dependencies.py.

"""
Operational notes
-----------------

Architecture prefixes are no longer hardcoded in this script.

To preserve the previous "all architectures under a root" behavior,
use automatic prefix discovery:

    python update_dependencies.py \
        --package torch \
        --auto-detect-prefixes \
        --base-prefix v2/

This dynamically discovers child prefixes such as:

    v2/gfx110X-all
    v2/gfx120X-all
    v2/gfx950-dcgpu

For narrow reconciliation of a single prefix, use:

    python update_dependencies.py \
        --package torch \
        --prefix v2/gfx110X-all
"""

from typing import Dict, List
from os import getenv

import re
import boto3  # type: ignore[import-untyped]
from boto3.resources.base import ServiceResource
from botocore.exceptions import ClientError

# Whitelist of allowed wheel platform and Python tags.
# Wheels not matching both criteria are skipped (not uploaded to S3).

# Exact platform tags that are always allowed.
_ALLOWED_PLATFORM_TAGS: frozenset[str] = frozenset(
    {
        "linux_x86_64",
        "win_amd64",  # Windows x64 — not excluded by the blacklist
        "any",  # pure-Python / platform-independent wheels
    }
)

# CPython version tags allowed for upload.
# Pure-Python wheels (python tag starting with "py") are also allowed
# regardless of version — they carry no CPython ABI dependency.
_ALLOWED_CPYTHON_TAGS: frozenset[str] = frozenset(
    {
        "cp310",
        "cp311",
        "cp312",
        "cp313",
        "cp314",
    }
)

PACKAGES_PER_PROJECT = {
    "dbus_python": {"versions": ["latest"], "project": "jax"},
    "flatbuffers": {"versions": ["latest"], "project": "jax"},
    "ml_dtypes": {"versions": ["latest"], "project": "jax"},
    "opt_einsum": {"versions": ["latest"], "project": "jax"},
    "tomli": {"versions": ["latest"], "project": "jax"},
    "sympy": {"versions": ["latest"], "project": "torch"},
    "mpmath": {"versions": ["1.3.0"], "project": "torch"},
    "pillow": {"versions": ["latest"], "project": "torch"},
    # 3.4.2 for Python 3.10, latest for Python 3.11+
    "networkx": {"versions": ["3.4.2", "latest"], "project": "torch"},
    "numpy": {"versions": ["latest"], "project": "torch"},
    "jinja2": {"versions": ["latest"], "project": "torch"},
    "markupsafe": {"versions": ["latest"], "project": "torch"},
    "filelock": {"versions": ["latest"], "project": "torch"},
    "fsspec": {"versions": ["latest"], "project": "torch"},
    "typing-extensions": {"versions": ["latest"], "project": "torch"},
    "rocm-bootstrap": {"versions": ["latest"], "project": "rocm"},
    "setuptools": {"versions": ["81.0.0"], "project": "rocm"},
}


# Product-local index names for structured publishing. All mirrored
# dependencies land under the "core" product, keyed by one of these indexes.
_STRUCTURED_INDEX_NAMES: frozenset[str] = frozenset({"whl", "whl-next"})
_STRUCTURED_PRODUCT = "core"


def normalize_package_name(name: str) -> str:
    """Normalize a Python distribution name for comparison."""
    return re.sub(r"[-_.]+", "-", name).lower()


def structured_dependency_key(index: str, pkg_name: str, filename: str) -> str:
    """Return the structured product-local S3 key for a dependency wheel.

    The package directory is derived from ``pkg_name`` (normalized to the
    canonical dashed form); ``filename`` is the wheel basename and keeps
    its original PEP 427 escaping (underscores).

    Example:
        structured_dependency_key("whl", "ml_dtypes", "ml_dtypes-0.5.0-...whl")
        -> "v5/core/whl/ml-dtypes/ml_dtypes-0.5.0-...whl"
    """
    if index not in _STRUCTURED_INDEX_NAMES:
        raise ValueError(
            f"index={index!r} is invalid; "
            f"expected one of {sorted(_STRUCTURED_INDEX_NAMES)}"
        )
    package_dir = normalize_package_name(pkg_name)
    return f"v5/{_STRUCTURED_PRODUCT}/{index}/{package_dir}/{filename}"


def get_selected_packages(
    *,
    package: str = "torch",
    dependency_names: frozenset[str] | None = None,
) -> dict[str, dict]:
    """Select entries from PACKAGES_PER_PROJECT for a project (or all).

    ``package`` is a project name (e.g. "torch") or "all" to span every
    project. ``dependency_names`` optionally narrows the result to specific
    package names; unknown names raise ValueError.
    """
    if package != "all" and package not in get_project_paths():
        raise ValueError(
            f"Unsupported package '{package}'. Expected 'all' or one of: "
            f"{', '.join(get_project_paths())}"
        )

    project_packages = {
        pkg_name: pkg_info
        for pkg_name, pkg_info in PACKAGES_PER_PROJECT.items()
        if package == "all" or pkg_info["project"] == package
    }

    if dependency_names is None:
        return project_packages

    normalized_dependency_names = frozenset(
        normalize_package_name(name) for name in dependency_names
    )
    available_names = frozenset(
        normalize_package_name(name) for name in project_packages
    )
    unmatched = normalized_dependency_names - available_names
    if unmatched:
        raise ValueError(
            f"Unknown --dependency-package value(s) for '{package}': "
            f"{sorted(unmatched)}. Valid names: {sorted(available_names)}"
        )

    return {
        pkg_name: pkg_info
        for pkg_name, pkg_info in project_packages.items()
        if normalize_package_name(pkg_name) in normalized_dependency_names
    }


def get_project_paths() -> List[str]:
    # Deduplicate project names from PACKAGES_PER_PROJECT and return them sorted.
    return sorted(
        set(pkg_info["project"] for pkg_info in PACKAGES_PER_PROJECT.values())
    )


def get_dependency_package_names(project: str) -> frozenset[str]:
    """
    Return dependency package names for the given project.

    Used by Lambda-side dependency trigger filtering.
    """
    return frozenset(
        pkg_name
        for pkg_name, pkg_info in PACKAGES_PER_PROJECT.items()
        if pkg_info["project"] == project
    )


def get_s3_bucket(bucket_name: str | None = None) -> ServiceResource:
    s3 = boto3.resource("s3")
    resolved_bucket_name = bucket_name or getenv("S3_BUCKET_PY")
    if not resolved_bucket_name:
        raise RuntimeError("Bucket must be provided via --bucket or S3_BUCKET_PY")
    return s3.Bucket(resolved_bucket_name)


def detect_prefixes_from_bucket(bucket: ServiceResource, base_prefix: str) -> List[str]:
    normalized_base_prefix = base_prefix.rstrip("/") + "/"
    print(f"INFO: Auto-detecting prefixes under '{normalized_base_prefix}'")

    # Reuse the bucket-associated client/session.
    client = bucket.meta.client
    paginator = client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(
        Bucket=bucket.name,
        Prefix=normalized_base_prefix,
        Delimiter="/",
    )

    prefixes: set[str] = set()
    for page in page_iterator:
        for common_prefix in page.get("CommonPrefixes", []):
            prefixes.add(common_prefix["Prefix"].rstrip("/"))

    detected_prefixes = sorted(prefixes)
    print(f"INFO: Detected prefixes: {detected_prefixes}")
    return detected_prefixes


def resolve_target_prefixes(
    *,
    bucket: ServiceResource,
    explicit_prefix: str | None = None,
    auto_detect_prefixes: bool = False,
    base_prefix: str | None = None,
) -> List[str]:
    if explicit_prefix:
        return [explicit_prefix.rstrip("/")]

    if base_prefix and not auto_detect_prefixes:
        raise RuntimeError(
            "--auto-detect-prefixes must be provided when using --base-prefix"
        )

    if auto_detect_prefixes:
        if not base_prefix:
            raise RuntimeError(
                "--base-prefix must be provided when using --auto-detect-prefixes"
            )
        return detect_prefixes_from_bucket(bucket, base_prefix)

    raise RuntimeError(
        "Must provide either --prefix or --auto-detect-prefixes with --base-prefix"
    )


def download(url: str) -> bytes:
    from urllib.request import urlopen

    with urlopen(url) as conn:
        return conn.read()


def is_stable(package_version: str) -> bool:
    return bool(re.match(r"^([0-9]+\.)+[0-9]+$", package_version))


def parse_simple_idx(url: str) -> Dict[str, str]:
    html = download(url).decode("ascii")
    return {
        name: url
        for (url, name) in re.findall('<a href="([^"]+)"[^>]*>([^>]+)</a>', html)
    }


def get_whl_versions(idx: Dict[str, str]) -> List[str]:
    return [
        k.split("-")[1]
        for k in idx.keys()
        if k.endswith(".whl") and is_stable(k.split("-")[1])
    ]


def get_wheels_of_version(idx: Dict[str, str], version: str) -> Dict[str, str]:
    return {
        k: v
        for (k, v) in idx.items()
        if k.endswith(".whl") and k.split("-")[1] == version
    }


def is_wheel_allowed(pkg: str) -> bool:
    """Return True if this wheel filename should be uploaded to S3.

    Both criteria must be satisfied:
    1. Platform tag is "linux_x86_64", "win_amd64", "any", or starts with
       "manylinux" and ends with "_x86_64" (e.g., "manylinux_2_17_x86_64").
       This rejects win32, win_arm64, macOS, musllinux, ARM, RISC-V, iOS, etc.
    2. Python tag is in _ALLOWED_CPYTHON_TAGS, or is exactly "py3"
       (pure-Python wheels). This rejects PyPy (pp*), cp39, cp313t,
       cp314t, py2, py2.py3, etc.

    Per PEP 427, the wheel stem is:
        {name}-{version}[-{build}]-{python}-{abi}-{platform}
    The last three fields are always python, abi, platform — regardless of
    whether the optional build tag is present.
    """
    if not pkg.endswith(".whl"):
        return False
    parts = pkg[:-4].split("-")
    if len(parts) < 5:
        return False  # Malformed — skip rather than guess

    platform_tag = parts[-1]
    python_tag = parts[-3]

    platform_ok = platform_tag in _ALLOWED_PLATFORM_TAGS or (
        platform_tag.startswith("manylinux") and platform_tag.endswith("_x86_64")
    )
    python_ok = python_tag in _ALLOWED_CPYTHON_TAGS or python_tag == "py3"

    return platform_ok and python_ok


def s3_object_exists(bucket: ServiceResource, key: str) -> bool:
    try:
        bucket.meta.client.head_object(Bucket=bucket.name, Key=key)
        return True
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def upload_missing_whls(
    bucket: ServiceResource,
    pkg_name: str = "numpy",
    prefix: str = "whl/test",
    *,
    dry_run: bool = False,
    only_pypi: bool = False,
    target_version: str = "latest",
    structured: bool = False,
    index: str = "whl",
) -> None:
    pypi_idx = parse_simple_idx(f"https://pypi.org/simple/{pkg_name}")
    pypi_versions = get_whl_versions(pypi_idx)

    # Determine which version to use
    if target_version == "latest" or not target_version:
        selected_version = pypi_versions[-1] if pypi_versions else None
    elif target_version in pypi_versions:
        selected_version = target_version
    else:
        print(
            f"Warning: Version {target_version} not found for {pkg_name}, using latest"
        )
        selected_version = pypi_versions[-1] if pypi_versions else None

    if not selected_version:
        print(f"No stable versions found for {pkg_name}")
        return

    pypi_latest_packages = get_wheels_of_version(pypi_idx, selected_version)

    # if not only_pypi:
    #     download_idx = parse_simple_idx(
    #         f"https://download.pytorch.org/{prefix}/{pkg_name}"
    #     )

    # Destination directory for status messages. In structured mode the flat
    # prefix is unused, so report the actual package directory instead.
    if structured:
        location = f"{_STRUCTURED_PRODUCT}/{index}/{normalize_package_name(pkg_name)}"
    else:
        location = prefix

    has_updates = False
    uploaded_or_present = 0

    for pkg in pypi_latest_packages:
        if not is_wheel_allowed(pkg):
            continue

        if structured:
            s3_key = structured_dependency_key(index, pkg_name, pkg)
        else:
            s3_key = f"{prefix}/{pkg}"
        dest_dir = s3_key.rsplit("/", 1)[0]
        if s3_object_exists(bucket, s3_key):
            print(f"Skipping existing {pkg} at s3://{bucket.name}/{s3_key}")
            uploaded_or_present += 1
            continue

        print(f"Downloading {pkg}")
        if dry_run:
            has_updates = True
            uploaded_or_present += 1
            print(f"Dry Run - not Uploading {pkg} to s3://{bucket.name}/{dest_dir}/")
            continue

        data = download(pypi_idx[pkg])
        print(f"Uploading {pkg} to s3://{bucket.name}/{dest_dir}/")
        bucket.Object(key=s3_key).put(ContentType="binary/octet-stream", Body=data)
        has_updates = True
        uploaded_or_present += 1

    if uploaded_or_present == 0:
        print(
            f"No allowed wheels found for {pkg_name} version {selected_version} "
            f"for {location}"
        )
    elif not has_updates:
        print(
            f"{pkg_name} is already at latest version {selected_version} "
            f"for {location}"
        )


def run_update_dependencies(
    *,
    package: str = "torch",
    dry_run: bool = False,
    only_pypi: bool = False,
    bucket_name: str | None = None,
    prefix: str | None = None,
    auto_detect_prefixes: bool = False,
    base_prefix: str | None = None,
    dependency_names: frozenset[str] | None = None,
    structured: bool = False,
    index: str = "whl",
) -> None:
    print(
        f"Running update_dependencies for package={package}, "
        f"structured={structured}, dry_run={dry_run}"
    )

    selected_packages = get_selected_packages(
        package=package, dependency_names=dependency_names
    )
    if not selected_packages:
        raise ValueError(f"No dependency packages selected for '{package}'")

    bucket = get_s3_bucket(bucket_name)

    if structured:
        _run_structured(
            bucket=bucket,
            selected_packages=selected_packages,
            index=index,
            dry_run=dry_run,
            only_pypi=only_pypi,
        )
        return

    target_prefixes = resolve_target_prefixes(
        bucket=bucket,
        explicit_prefix=prefix,
        auto_detect_prefixes=auto_detect_prefixes,
        base_prefix=base_prefix,
    )

    for full_path in target_prefixes:
        for pkg_name, pkg_info in selected_packages.items():
            pkg_prefix = full_path
            if "target" in pkg_info and pkg_info["target"] != "":
                pkg_prefix = f"{full_path}/{pkg_info['target']}"

            for target_version in pkg_info["versions"]:
                upload_missing_whls(
                    bucket,
                    pkg_name,
                    pkg_prefix,
                    dry_run=dry_run,
                    only_pypi=only_pypi,
                    target_version=target_version,
                )


def _run_structured(
    *,
    bucket: ServiceResource,
    selected_packages: dict[str, dict],
    index: str,
    dry_run: bool,
    only_pypi: bool,
) -> None:
    """Mirror selected dependencies into core/<index>/<package>/ directories.

    The structured layout is not prefix-driven: each wheel's destination key is
    computed from its package name, so there is no prefix fan-out.
    """
    if index not in _STRUCTURED_INDEX_NAMES:
        raise ValueError(
            f"index={index!r} is invalid; "
            f"expected one of {sorted(_STRUCTURED_INDEX_NAMES)}"
        )
    for pkg_name, pkg_info in selected_packages.items():
        for target_version in pkg_info["versions"]:
            upload_missing_whls(
                bucket,
                pkg_name,
                dry_run=dry_run,
                only_pypi=only_pypi,
                target_version=target_version,
                structured=True,
                index=index,
            )


def main() -> None:
    from argparse import ArgumentParser

    parser = ArgumentParser("Upload dependent packages to S3")
    project_paths = get_project_paths()
    parser.add_argument(
        "--package",
        choices=[*project_paths, "all"],
        default=None,
        help=(
            "Project whose dependencies to mirror, or 'all' for every "
            "project. Default: 'all' in structured mode, 'torch' otherwise."
        ),
    )
    parser.add_argument(
        "--structured",
        action="store_true",
        help=(
            "Mirror dependencies into product-local package directories "
            "(core/<index>/<package>/) instead of a flat prefix. Incompatible "
            "with --prefix, --auto-detect-prefixes, and --base-prefix."
        ),
    )
    parser.add_argument(
        "--index",
        default="whl",
        choices=["whl", "whl-next"],
        help=(
            "Product-local index name for structured publishing (default: "
            "whl). Selects the core/<index>/ path segment."
        ),
    )
    parser.add_argument("--bucket", type=str, help="S3 bucket name")
    parser.add_argument(
        "--prefix",
        type=str,
        help=(
            "Explicit prefix to update "
            "(e.g. v2/gfx110X-all, v2-staging/gfx110X-all, v4/whl)"
        ),
    )
    parser.add_argument(
        "--auto-detect-prefixes",
        action="store_true",
        help=(
            "Automatically detect architecture prefixes under the given base "
            "path using S3 CommonPrefixes."
        ),
    )
    parser.add_argument(
        "--base-prefix",
        type=str,
        help=(
            "Base prefix for auto-detection (e.g. v2/, v2-staging/, v3/). "
            "Required when using --auto-detect-prefixes."
        ),
    )
    parser.add_argument(
        "--dependency-package",
        action="append",
        dest="dependency_packages",
        help=(
            "Limit reconciliation to one dependency package. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-pypi", action="store_true")

    args = parser.parse_args()

    if args.structured:
        conflicting = [
            name
            for name, value in (
                ("--prefix", args.prefix),
                ("--auto-detect-prefixes", args.auto_detect_prefixes),
                ("--base-prefix", args.base_prefix),
            )
            if value
        ]
        if conflicting:
            parser.error(
                f"--structured is incompatible with flat-only flag(s): "
                f"{', '.join(conflicting)}"
            )

    # Default package depends on mode: structured mirrors everything by default;
    # flat keeps the historical single-project ("torch") default.
    package = args.package
    if package is None:
        package = "all" if args.structured else "torch"

    run_update_dependencies(
        package=package,
        dry_run=args.dry_run,
        only_pypi=args.only_pypi,
        bucket_name=args.bucket,
        prefix=args.prefix,
        auto_detect_prefixes=args.auto_detect_prefixes,
        base_prefix=args.base_prefix,
        dependency_names=(
            frozenset(args.dependency_packages) if args.dependency_packages else None
        ),
        structured=args.structured,
        index=args.index,
    )


if __name__ == "__main__":
    main()
