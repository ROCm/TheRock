# Copyright Facebook, Inc. and its affiliates.
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# Forked from https://github.com/pytorch/test-infra/blob/1ffc7f7b3b421b57c380de469e11744f54399f09/s3_management/update_dependencies.py.
# Changes incorporated from https://github.com/pytorch/test-infra/blob/a87d94b148bbd2c68e69e542350099a971f4c8d3/s3_management/update_dependencies.py.

from typing import Dict, List
from os import getenv

import boto3  # type: ignore[import-untyped]
import re


S3 = boto3.resource("s3")
CLIENT = boto3.client("s3")
# We also manage `therock-nightly-python` (not the default to make the script safer to test)
BUCKET = S3.Bucket(getenv("S3_BUCKET_PY", "therock-dev-python"))
# Note: v2-staging first, in case issues are observed while the script runs
# and the developer wants to more safely cancel the script.
VERSIONS = ["v2-staging", "v2"]

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
    "rocm-bootstrap": {"versions": ["latest"], "project": "torch"},
    "setuptools": {"versions": ["81.0.0"], "project": "rocm"},
}


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
       cp314, cp314t, py2, py2.py3, etc.

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


def upload_missing_whls(
    pkg_name: str = "numpy",
    prefix: str = "whl/test",
    *,
    dry_run: bool = False,
    only_pypi: bool = False,
    target_version: str = "latest",
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

    download_latest_packages: Dict[str, str] = {}
    # if not only_pypi:
    #     download_idx = parse_simple_idx(
    #         f"https://download.pytorch.org/{prefix}/{pkg_name}"
    #     )

    has_updates = False
    for pkg in pypi_latest_packages:
        if pkg in download_latest_packages:
            continue
        if not is_wheel_allowed(pkg):
            continue
        print(f"Downloading {pkg}")
        if dry_run:
            has_updates = True
            print(f"Dry Run - not Uploading {pkg} to s3://{BUCKET.name}/{prefix}/")
            continue
        data = download(pypi_idx[pkg])
        print(f"Uploading {pkg} to s3://{BUCKET.name}/{prefix}/")
        BUCKET.Object(key=f"{prefix}/{pkg}").put(
            ContentType="binary/octet-stream", Body=data
        )
        has_updates = True
    if not has_updates:
        print(
            f"{pkg_name} is already at latest version {selected_version} for {prefix}"
        )


def main() -> None:
    from argparse import ArgumentParser

    parser = ArgumentParser(f"Upload dependent packages to s3://{BUCKET}")
    # Get unique paths from the packages list
    project_paths = list(
        set(pkg_info["project"] for pkg_info in PACKAGES_PER_PROJECT.values())
    )
    parser.add_argument("--package", choices=project_paths, default="torch")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-pypi", action="store_true")
    args = parser.parse_args()

    SUBFOLDERS = [
        "gfx101X-dgpu",
        "gfx103X-all",
        "gfx110X-all",
        "gfx1150",
        "gfx1151",
        "gfx120X-all",
        "gfx90X-dcgpu",
        "gfx94X-dcgpu",
        "gfx950-dcgpu",
    ]

    for prefix in SUBFOLDERS:
        # Filter packages by the selected project path
        selected_packages = {
            pkg_name: pkg_info
            for pkg_name, pkg_info in PACKAGES_PER_PROJECT.items()
            if pkg_info["project"] == args.package
        }
        for VERSION in VERSIONS:
            for pkg_name, pkg_info in selected_packages.items():
                if "target" in pkg_info and pkg_info["target"] != "":
                    full_path = f'{VERSION}/{prefix}/{pkg_info["target"]}'
                else:
                    full_path = f"{VERSION}/{prefix}"

                for target_version in pkg_info["versions"]:
                    upload_missing_whls(
                        pkg_name,
                        full_path,
                        dry_run=args.dry_run,
                        only_pypi=args.only_pypi,
                        target_version=target_version,
                    )


if __name__ == "__main__":
    main()
