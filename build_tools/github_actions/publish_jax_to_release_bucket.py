#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Upload multi-arch JAX wheels from a local directory to a release bucket."""

import argparse
import logging
import sys
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from _therock_utils.s3_buckets import get_release_bucket_config
from _therock_utils.python_package_paths import plan_local_uploads
from _therock_utils.storage_backend import create_storage_backend
from _therock_utils.storage_location import StorageLocation
from github_actions.github_actions_api import gha_set_output

logger = logging.getLogger(__name__)

MULTI_ARCH_INDEX_URLS = {
    "dev": "https://rocm.devreleases.amd.com/whl-multi-arch/",
    "nightly": "https://rocm.nightlies.amd.com/whl-multi-arch/",
    "prerelease": "https://rocm.prereleases.amd.com/whl-multi-arch/",
}


def _publish_structured(source_dir, dest_bucket, index, backend) -> None:
    """Upload wheels into product-local package directories.

    Plans per-wheel destinations under ``v5/rocm/jax/<index>/<package>/``
    and uploads them, failing fast if the source directory holds no wheels.
    """
    plans = plan_local_uploads(source_dir, dest_bucket, "jax", index)
    if not plans:
        raise FileNotFoundError(f"No wheels found at {source_dir}")
    for plan in plans:
        logger.info("JAX wheel: %s -> %s", plan.source, plan.dest.s3_uri)
    count = backend.upload_files([(plan.source, plan.dest) for plan in plans])
    logger.info("Uploaded %d wheel files (structured)", count)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Upload multi-arch JAX wheels to a release bucket"
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        type=Path,
        help="Local directory containing the wheels to upload",
    )
    parser.add_argument(
        "--release-type",
        required=True,
        choices=["dev", "nightly", "prerelease"],
        help="Release type (selects therock-{release_type}-python bucket)",
    )
    parser.add_argument(
        "--structured",
        action="store_true",
        help="Publish wheels into product-local package directories "
        "(v5/rocm/jax/<index>/<package>/) instead of the flat v4/whl "
        "prefix.",
    )
    parser.add_argument(
        "--python-index",
        default="whl",
        choices=["whl", "whl-next"],
        help="Product-local index name for structured publishing (default: "
        "whl). Selects the v5/rocm/jax/<index>/ path segment.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan without uploading"
    )
    args = parser.parse_args(argv)

    if not args.source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {args.source_dir}")

    bucket = get_release_bucket_config(args.release_type, "python")
    backend = create_storage_backend(dry_run=args.dry_run)

    if args.structured:
        _publish_structured(args.source_dir, bucket.name, args.python_index, backend)
    else:
        dest = StorageLocation(bucket.name, "v4/whl")
        logger.info("JAX wheels: %s -> %s", args.source_dir, dest.s3_uri)
        count = backend.upload_directory(args.source_dir, dest, include=["*.whl"])
        logger.info("Uploaded %d wheel files", count)
        if count == 0:
            raise FileNotFoundError(f"No wheels found at {args.source_dir}")

    gha_set_output({"package_index_url": MULTI_ARCH_INDEX_URLS[args.release_type]})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
