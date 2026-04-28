#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Upload PyTorch wheels from a local directory to release-bucket staging.

Used by the multi-arch release PyTorch wheels workflow to push the
host wheel and per-gfx amd-torch-device-* wheels produced by the kpack
splitter into the release bucket's staging path.

Example with ``--source-dir /tmp/dist --release-type dev``::

    /tmp/dist/torch-2.10.0+rocm7.10.0-cp312-cp312-linux_x86_64.whl
    /tmp/dist/amd-torch-device-gfx942-2.10.0+rocm7.10.0-py3-none-linux_x86_64.whl
      -> s3://therock-dev-python/v4/whl-staging/torch-...whl
      -> s3://therock-dev-python/v4/whl-staging/amd-torch-device-...whl

Test usage::

    python build_tools/github_actions/publish_pytorch_to_staging.py \\
        --source-dir /tmp/dist --release-type dev --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from _therock_utils.s3_buckets import get_release_bucket_config
from _therock_utils.storage_backend import create_storage_backend
from _therock_utils.storage_location import StorageLocation

logger = logging.getLogger(__name__)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Upload PyTorch wheels to release-bucket staging"
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
        "--dry-run", action="store_true", help="Print plan without uploading"
    )
    args = parser.parse_args(argv)

    if not args.source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {args.source_dir}")

    bucket = get_release_bucket_config(args.release_type, "python")
    s3_subdir = "v4/whl-staging"
    dest = StorageLocation(bucket.name, s3_subdir)
    backend = create_storage_backend(dry_run=args.dry_run)

    logger.info("PyTorch wheels: %s -> %s", args.source_dir, dest.s3_uri)
    count = backend.upload_directory(args.source_dir, dest, include=["*.whl"])
    logger.info("Uploaded %d wheel files", count)
    if count == 0:
        raise FileNotFoundError(f"No wheels found at {args.source_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
