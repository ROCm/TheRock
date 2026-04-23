#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Publish ROCm release files from an artifacts bucket to release buckets.

These release file types are supported:

- [x] tarballs
- [ ] python packages
- [ ] native linux packages
- [ ] native windows packages

Example with ``--run-id 12345 --platform linux --release-type dev``:

    s3://therock-dev-artifacts/12345-linux/tarballs/therock-dist-linux-gfx94X-dcgpu-7.10.0.tar.gz
      -> s3://therock-dev-tarball/v4/tarball/therock-dist-linux-gfx94X-dcgpu-7.10.0.tar.gz

Test usage:
    python build_tools/github_actions/publish_rocm_to_release_buckets.py \\
        --run-id 12345 --platform linux --release-type dev --dry-run
"""

import argparse
import logging
import platform as platform_module
import sys
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from _therock_utils.s3_buckets import get_release_bucket_config
from _therock_utils.storage_backend import StorageBackend, create_storage_backend
from _therock_utils.storage_location import StorageLocation
from _therock_utils.workflow_outputs import WorkflowOutputRoot

logger = logging.getLogger(__name__)


def publish_tarballs(
    artifacts_root: WorkflowOutputRoot,
    release_type: str,
    backend: StorageBackend,
) -> int:
    """Copy tarballs from the artifacts bucket to the release tarball bucket.

    Example:
        s3://therock-dev-artifacts/12345-linux/tarballs/
          -> s3://therock-dev-tarball/v4/tarball/

    Returns:
        Number of tarballs copied.
    """
    source = artifacts_root.tarballs()
    dest_bucket = get_release_bucket_config(release_type, "tarball")
    dest = StorageLocation(dest_bucket.name, "v4/tarball")

    logger.info("Tarballs: %s -> %s", source.s3_uri, dest.s3_uri)
    count = backend.copy_directory(source, dest, include=["*.tar.gz"])
    logger.info("Copied %d tarballs", count)
    if count == 0:
        raise FileNotFoundError(f"No tarballs found at {source.s3_uri}")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Publish ROCm release files to release buckets"
    )
    parser.add_argument("--run-id", required=True, help="Source workflow run ID")
    parser.add_argument(
        "--platform",
        default=platform_module.system().lower(),
        choices=["linux", "windows"],
        help="Platform (default: current system)",
    )
    parser.add_argument(
        "--release-type",
        required=True,
        choices=["dev", "nightly", "prerelease"],
        help="Release type (determines source and destination buckets)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan without copying"
    )
    args = parser.parse_args(argv)

    artifacts_root = WorkflowOutputRoot.from_workflow_run(
        run_id=args.run_id, platform=args.platform, release_type=args.release_type
    )
    backend = create_storage_backend(dry_run=args.dry_run)

    publish_tarballs(artifacts_root, args.release_type, backend)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
