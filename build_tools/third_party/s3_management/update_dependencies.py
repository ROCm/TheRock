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

PACKAGES_PER_PROJECT = {
    "dbus_python": {"version": "latest", "project": "jax"},
    "flatbuffers": {"version": "latest", "project": "jax"},
    "ml_dtypes": {"version": "latest", "project": "jax"},
    "opt_einsum": {"version": "latest", "project": "jax"},
    "tomli": {"version": "latest", "project": "jax"},
    "sympy": {"version": "latest", "project": "torch"},
    "mpmath": {"version": "latest", "project": "torch"},
    "pillow": {"version": "latest", "project": "torch"},
    "networkx": {"version": "latest", "project": "torch"},
    "numpy": {"version": "latest", "project": "torch"},
    "jinja2": {"version": "latest", "project": "torch"},
    "filelock": {"version": "latest", "project": "torch"},
    "fsspec": {"version": "latest", "project": "torch"},
    "typing-extensions": {"version": "latest", "project": "torch"},
    "setuptools": {"version": "latest", "project": "rocm"},
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
        # Skip pp packages
        if "-pp3" in pkg:
            continue
        # Skip win32 packages
        if "-win32" in pkg:
            continue
        # Skip win_arm64 packages
        if "-win_arm64" in pkg:
            continue
        # Skip muslinux packages
        if "-musllinux" in pkg:
            continue
        # Skip macosx packages
        if "-macosx" in pkg:
            continue
        # Skip aarch64 packages
        if "aarch64" in pkg:
            continue
        # Skip i686 packages
        if "i686" in pkg:
            continue
        # Skip unsupported Python version
        if "cp39" in pkg:
            continue
        if "cp310" in pkg:
            continue
        if "cp313t" in pkg:
            continue
        if "cp314" in pkg:
            continue
        if "cp314t" in pkg:
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

    SUBFOLDERS =  [
        "gfx110X-dgpu",
        "gfx1151",
        "gfx120X-all",
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

                upload_missing_whls(
                    pkg_name,
                    full_path,
                    dry_run=args.dry_run,
                    only_pypi=args.only_pypi,
                    target_version=pkg_info["version"],
                )


if __name__ == "__main__":
    main()
