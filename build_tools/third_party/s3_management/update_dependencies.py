# Copyright Facebook, Inc. and its affiliates.
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# Forked from https://github.com/pytorch/test-infra/blob/1ffc7f7b3b421b57c380de469e11744f54399f09/s3_management/update_dependencies.py

from typing import Dict, List

import boto3
import re


S3 = boto3.resource("s3")
CLIENT = boto3.client("s3")
BUCKET = S3.Bucket("therock-nightly-python")
VERSION = "v2"

PACKAGES_PER_PROJECT = {
    "jax": [
        "dbus-python",
        "flatbuffers",
        "ml_dtypes",
        "opt_einsum",
        "tomli",
    ],
    "torch": [
        "sympy",
        "mpmath",
        "pillow",
        "networkx",
        "numpy",
        "jinja2",
        "MarkupSafe",
        "filelock",
        "fsspec",
        "typing-extensions",
    ],
    "rocm": [
        "setuptools",
    ],
}


def download(url: str) -> bytes:
    from urllib.request import urlopen

    with urlopen(url) as conn:
        return conn.read()


def is_stable(package_version: str) -> bool:
    return bool(re.match(r'^([0-9]+\.)+[0-9]+$', package_version))


def parse_simple_idx(url: str) -> Dict[str, str]:
    html = download(url).decode("ascii")
    return {
        name: url
        for (url, name) in re.findall('<a href="([^"]+)"[^>]*>([^>]+)</a>', html)
    }


def get_whl_versions(idx: Dict[str, str]) -> List[str]:
    return [k.split("-")[1] for k in idx.keys() if k.endswith(".whl") and is_stable(k.split("-")[1])]


def get_wheels_of_version(idx: Dict[str, str], version: str) -> Dict[str, str]:
    return {
        k: v
        for (k, v) in idx.items()
        if k.endswith(".whl") and k.split("-")[1] == version
    }


def upload_missing_whls(
    pkg_name: str = "numpy", prefix: str = "whl/test", *, dry_run: bool = False, only_pypi: bool = False
) -> None:
    pypi_idx = parse_simple_idx(f"https://pypi.org/simple/{pkg_name}")
    pypi_versions = get_whl_versions(pypi_idx)
    pypi_latest_packages = get_wheels_of_version(pypi_idx, pypi_versions[-1])

    download_latest_packages = []
    # if not only_pypi:
    #     download_idx = parse_simple_idx(f"https://download.pytorch.org/{prefix}/{pkg_name}")
    #     download_latest_packages = get_wheels_of_version(download_idx, pypi_versions[-1])

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
        print(f"Downloading {pkg}")
        if dry_run:
            has_updates = True
            continue
        data = download(pypi_idx[pkg])
        print(f"Uploading {pkg} to s3://{BUCKET}/{prefix}/")
        BUCKET.Object(key=f"{prefix}/{pkg}").put(
            ContentType="binary/octet-stream", Body=data
        )
        has_updates = True
    if not has_updates:
        print(
            f"{pkg_name} is already at latest version {pypi_versions[-1]} for {prefix}"
        )


def main() -> None:
    from argparse import ArgumentParser

    parser = ArgumentParser(f"Upload dependent packages to s3://{BUCKET}")
    parser.add_argument("--package", choices=PACKAGES_PER_PROJECT.keys(), default="torch")
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
        for package in PACKAGES_PER_PROJECT[args.package]:
            upload_missing_whls(package, f"{VERSION}/{prefix}", dry_run=args.dry_run, only_pypi=args.only_pypi)


if __name__ == "__main__":
    main()
