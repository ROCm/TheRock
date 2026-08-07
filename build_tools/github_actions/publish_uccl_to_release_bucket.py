#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Upload multi-architecture UCCL wheels to a TheRock release bucket."""

import argparse
import logging
from pathlib import Path
import sys


_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from _therock_utils.s3_buckets import get_release_bucket_config
from _therock_utils.storage_backend import create_storage_backend
from _therock_utils.storage_location import StorageLocation
from github_actions.github_actions_api import gha_set_output


logger = logging.getLogger(__name__)

MULTI_ARCH_INDEX_URLS = {
    "dev": "https://rocm.devreleases.amd.com/whl-multi-arch/",
    "nightly": "https://rocm.nightlies.amd.com/whl-multi-arch/",
    "prerelease": "https://rocm.prereleases.amd.com/whl-multi-arch/",
}


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Upload multi-architecture UCCL wheels to a release bucket"
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        type=Path,
        help="Local directory containing the UCCL wheel",
    )
    parser.add_argument(
        "--release-type",
        required=True,
        choices=["dev", "nightly", "prerelease"],
        help="Release type selecting the destination Python bucket",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the upload plan without modifying S3",
    )
    args = parser.parse_args(argv)

    if not args.source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {args.source_dir}")

    bucket = get_release_bucket_config(args.release_type, "python")
    destination = StorageLocation(bucket.name, "v4/whl")
    backend = create_storage_backend(dry_run=args.dry_run)

    logger.info("UCCL wheels: %s -> %s", args.source_dir, destination.s3_uri)
    count = backend.upload_directory(
        args.source_dir,
        destination,
        include=["uccl-*.whl"],
    )
    if count == 0:
        raise FileNotFoundError(f"No UCCL wheels found at {args.source_dir}")
    logger.info("Uploaded %d UCCL wheel files", count)

    gha_set_output({"package_index_url": MULTI_ARCH_INDEX_URLS[args.release_type]})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
