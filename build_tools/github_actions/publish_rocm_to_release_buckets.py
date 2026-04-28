#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Publish ROCm release files from an artifacts bucket to release buckets.

These release file types are supported:

- [x] tarballs
- [x] python packages
- [ ] native linux packages
- [ ] native windows packages

Example with ``--run-id 12345 --platform linux --release-type dev``:

    tarballs:

    s3://therock-dev-artifacts/12345-linux/tarballs/therock-dist-linux-gfx94X-dcgpu-7.10.0.tar.gz
      -> s3://therock-dev-tarball/v4/tarball/therock-dist-linux-gfx94X-dcgpu-7.10.0.tar.gz

    python (kpack split enabled):

    s3://therock-dev-artifacts/12345-linux/python/rocm-7.13.0.tar.gz
    s3://therock-dev-artifacts/12345-linux/python/rocm_sdk_core-7.13.0-py3-none-linux_x86_64.whl
    s3://therock-dev-artifacts/12345-linux/python/rocm_sdk_device_gfx1100-7.13.0-py3-none-linux_x86_64.whl
    s3://therock-dev-artifacts/12345-linux/python/rocm_sdk_libraries-7.13.0-py3-none-linux_x86_64.whl
      -> s3://therock-dev-python/v4/whl-staging/rocm-7.13.0.tar.gz
      -> s3://therock-dev-python/v4/whl-staging/rocm_sdk_core-7.13.0-py3-none-linux_x86_64.whl
      -> s3://therock-dev-python/v4/whl-staging/rocm_sdk_device_gfx1100-7.13.0-py3-none-linux_x86_64.whl
      -> s3://therock-dev-python/v4/whl-staging/rocm_sdk_libraries-7.13.0-py3-none-linux_x86_64.whl
      -> s3://therock-dev-python/v4/whl/rocm-7.13.0.tar.gz
      -> s3://therock-dev-python/v4/whl/rocm_sdk_core-7.13.0-py3-none-linux_x86_64.whl
      -> s3://therock-dev-python/v4/whl/rocm_sdk_device_gfx1100-7.13.0-py3-none-linux_x86_64.whl
      -> s3://therock-dev-python/v4/whl/rocm_sdk_libraries-7.13.0-py3-none-linux_x86_64.whl

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


def publish_python_packages(
    artifacts_root: WorkflowOutputRoot,
    release_type: str,
    backend: StorageBackend,
    kpack_split: bool,
) -> None:
    """Copy python packages from the artifacts bucket to the release python bucket.

    Wheels always land in both the -staging index (canonical superset) and
    the release index (current promoted set). The release path is treated as
    a subset of -staging, so anything visible from the release URL is also
    visible from the staging URL. A future test-gated promotion step would
    move the second copy out of this script.

    The destination layout depends on kpack_split:
      - kpack_split=False uses the v3 per-family layout (v3/whl-staging,
        v3/whl).
      - kpack_split=True uses the v4 flat layout (v4/whl-staging, v4/whl).

    Examples:

        kpack split disabled (per-family subdirs):
        s3://therock-dev-artifacts/12345-linux/python/gfx110X-all/*.whl
          -> s3://therock-dev-python/v3/whl-staging/gfx110X-all/*.whl
          -> s3://therock-dev-python/v3/whl/gfx110X-all/*.whl

        kpack split enabled (flat):
        s3://therock-dev-artifacts/12345-linux/python/*.whl
          -> s3://therock-dev-python/v4/whl-staging/*.whl
          -> s3://therock-dev-python/v4/whl/*.whl
    """
    source = artifacts_root.python_packages()
    dest_bucket = get_release_bucket_config(release_type, "python")
    release_subdir = "v4/whl" if kpack_split else "v3/whl"
    s3_subdirs = [f"{release_subdir}-staging", release_subdir]

    for s3_subdir in s3_subdirs:
        dest = StorageLocation(dest_bucket.name, s3_subdir)
        logger.info("Python packages: %s -> %s", source.s3_uri, dest.s3_uri)
        count = backend.copy_directory(source, dest, include=["*.whl", "*.tar.gz"])
        logger.info("Copied %d python package files to %s", count, s3_subdir)
        if count == 0:
            raise FileNotFoundError(f"No python packages found at {source.s3_uri}")


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
    # String "true"/"false" because GitHub Actions outputs are strings.
    parser.add_argument(
        "--kpack-split",
        default="false",
        help='Whether kpack split is enabled ("true" or "false")',
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan without copying"
    )
    args = parser.parse_args(argv)

    artifacts_root = WorkflowOutputRoot.from_workflow_run(
        run_id=args.run_id, platform=args.platform, release_type=args.release_type
    )
    backend = create_storage_backend(dry_run=args.dry_run)
    kpack_split = args.kpack_split.lower() == "true"

    publish_tarballs(artifacts_root, args.release_type, backend)
    publish_python_packages(artifacts_root, args.release_type, backend, kpack_split)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
