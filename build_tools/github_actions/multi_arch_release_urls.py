#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Single source of truth for multi-arch release package index URLs.

Maps a release type to the public pip index that serves its multi-arch
PyTorch wheels. Used both when publishing wheels and when dispatching the
full test suite so tests install from the same index the wheels were
published to.
"""

import argparse
import sys
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from github_actions.github_actions_api import gha_set_output

MULTI_ARCH_INDEX_URLS = {
    # TODO: Move this release bucket to CDN/index URL mapping into
    # build_tools/_therock_utils/s3_buckets.py.
    "dev": "https://rocm.devreleases.amd.com/whl-multi-arch/",
    "nightly": "https://rocm.nightlies.amd.com/whl-multi-arch/",
    "prerelease": "https://rocm.prereleases.amd.com/whl-multi-arch/",
}


def get_index_url(release_type: str) -> str:
    try:
        return MULTI_ARCH_INDEX_URLS[release_type]
    except KeyError:
        raise ValueError(
            f"Unknown release_type {release_type!r}; expected one of "
            f"{sorted(MULTI_ARCH_INDEX_URLS)}"
        )


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-type",
        required=True,
        choices=sorted(MULTI_ARCH_INDEX_URLS),
        help="Release type whose multi-arch index URL to resolve.",
    )
    args = parser.parse_args(argv)
    url = get_index_url(args.release_type)
    gha_set_output({"package_index_url": url})
    print(url)


if __name__ == "__main__":
    main(sys.argv[1:])
