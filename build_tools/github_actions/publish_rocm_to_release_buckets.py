#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Publish ROCm release files from an artifacts bucket to release buckets.

These release file types are supported:

- [x] tarballs
- [x] python packages
- [x] native linux packages
- [ ] native windows packages

Example with ``--run-id 12345 --platform linux --release-type dev``:

    tarballs:

    s3://therock-dev-artifacts/12345-linux/tarballs/therock-dist-linux-gfx94X-dcgpu-7.10.0.tar.gz
      -> s3://therock-dev-tarball/v4/tarball/therock-dist-linux-gfx94X-dcgpu-7.10.0.tar.gz

    python (kpack split enabled, multi-arch):

    s3://therock-dev-artifacts/12345-linux/python/rocm-7.13.0.tar.gz
    s3://therock-dev-artifacts/12345-linux/python/rocm_sdk_core-7.13.0-py3-none-linux_x86_64.whl
    s3://therock-dev-artifacts/12345-linux/python/rocm_sdk_device_gfx1100-7.13.0-py3-none-linux_x86_64.whl
    s3://therock-dev-artifacts/12345-linux/python/rocm_sdk_libraries-7.13.0-py3-none-linux_x86_64.whl
      -> s3://therock-dev-python/v4/whl/rocm-7.13.0.tar.gz
      -> s3://therock-dev-python/v4/whl/rocm_sdk_core-7.13.0-py3-none-linux_x86_64.whl
      -> s3://therock-dev-python/v4/whl/rocm_sdk_device_gfx1100-7.13.0-py3-none-linux_x86_64.whl
      -> s3://therock-dev-python/v4/whl/rocm_sdk_libraries-7.13.0-py3-none-linux_x86_64.whl

    native linux packages (dev/nightly):

    s3://therock-dev-artifacts/12345-linux/packages/deb/
      -> s3://therock-dev-packages/v4/deb/20250101-12345/
    s3://therock-dev-artifacts/12345-linux/packages/rpm/
      -> s3://therock-dev-packages/v4/rpm/20250101-12345/

    native linux packages (prerelease):

    s3://therock-prerelease-artifacts/12345-linux/packages/deb/
      -> s3://therock-prerelease-packages/v4/packages/deb/
    s3://therock-prerelease-artifacts/12345-linux/packages/rpm/
      -> s3://therock-prerelease-packages/v4/packages/rpm/

ASAN build variant:

    For ASAN builds (--build-variant asan), python packages are skipped and
    tarballs/native packages are published to separate paths:

    s3://therock-dev-artifacts/12345-linux/tarballs/
      -> s3://therock-dev-tarball/v4/tarball-asan/
    s3://therock-dev-artifacts/12345-linux/packages/deb/
      -> s3://therock-dev-packages/v4/packages-asan/deb/20250101-12345/
    s3://therock-dev-artifacts/12345-linux/packages/rpm/
      -> s3://therock-dev-packages/v4/packages-asan/rpm/20250101-12345/

Test usage:
    python build_tools/github_actions/publish_rocm_to_release_buckets.py \\
        --run-id 12345 --platform linux --release-type dev --dry-run
"""

import argparse
import datetime
import logging
import platform as platform_module
import sys
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from _therock_utils.s3_buckets import get_release_bucket_config
from _therock_utils.storage_backend import StorageBackend, create_storage_backend
from _therock_utils.storage_location import StorageLocation
from _therock_utils.python_package_paths import plan_key_copies
from _therock_utils.workflow_outputs import WorkflowOutputRoot

logger = logging.getLogger(__name__)


def publish_tarballs(
    artifacts_root: WorkflowOutputRoot,
    release_type: str,
    backend: StorageBackend,
    build_variant: str = "release",
) -> int:
    """Copy tarballs from the artifacts bucket to the release tarball bucket.

    Example:
        s3://therock-dev-artifacts/12345-linux/tarballs/
          -> s3://therock-dev-tarball/v4/tarball/

    ASAN example:
        s3://therock-dev-artifacts/12345-linux/tarballs/
          -> s3://therock-dev-tarball/v4/tarball-asan/

    Returns:
        Number of tarballs copied.
    """
    source = artifacts_root.tarballs()
    dest_bucket = get_release_bucket_config(release_type, "tarball")
    dest_path = "v4/tarball-asan" if build_variant == "asan" else "v4/tarball"
    dest = StorageLocation(dest_bucket.name, dest_path)

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
    structured: bool = False,
    index: str = "whl",
) -> None:
    """Copy python packages from the artifacts bucket to the release python bucket.

    The destination layout depends on kpack_split:
      - kpack_split=False (per-family): publishes to both v3/whl-staging and
        v3/whl. Test-gated promotion may later move wheels from staging to
        release separately.
      - kpack_split=True (multi-arch): publishes directly to v4/whl (no
        staging index — tests run post-publish as a signal, not a gate).

    Examples:

        kpack split disabled (per-family subdirs):
        s3://therock-dev-artifacts/12345-linux/python/gfx110X-all/*.whl
          -> s3://therock-dev-python/v3/whl-staging/gfx110X-all/*.whl
          -> s3://therock-dev-python/v3/whl/gfx110X-all/*.whl

        kpack split enabled (flat):
        s3://therock-dev-artifacts/12345-linux/python/*.whl
          -> s3://therock-dev-python/v4/whl/*.whl

    When ``structured`` is set (multi-arch only), packages are copied into
    product-local package directories instead of the flat v4/whl prefix:

        s3://therock-dev-artifacts/12345-linux/python/rocm_sdk_core-...whl
          -> s3://therock-dev-python/v5/rocm/core/<index>/rocm-sdk-core/rocm_sdk_core-...whl
    """
    source = artifacts_root.python_packages()
    dest_bucket = get_release_bucket_config(release_type, "python")

    if structured:
        _publish_python_packages_structured(
            source=source,
            dest_bucket=dest_bucket.name,
            backend=backend,
            index=index,
        )
        return

    if kpack_split:
        # Multi-arch: publish directly (no staging index).
        s3_subdirs = ["v4/whl"]
    else:
        # Per-family: publish to both staging and release.
        s3_subdirs = ["v3/whl-staging", "v3/whl"]

    for s3_subdir in s3_subdirs:
        dest = StorageLocation(dest_bucket.name, s3_subdir)
        logger.info("Python packages: %s -> %s", source.s3_uri, dest.s3_uri)
        count = backend.copy_directory(source, dest, include=["*.whl", "*.tar.gz"])
        logger.info("Copied %d python package files to %s", count, s3_subdir)
        if count == 0:
            raise FileNotFoundError(f"No python packages found at {source.s3_uri}")


def _publish_python_packages_structured(
    source: StorageLocation,
    dest_bucket: str,
    backend: StorageBackend,
    index: str,
) -> None:
    """Copy python packages into product-local package directories.

    Lists the flat source prefix, then copies each accepted artifact into
    ``v5/rocm/core/<index>/<normalized-package>/<filename>`` so the structured
    index generator can discover per-package directories.
    """
    source_objects = backend.list_files(source, include=["*.whl", "*.tar.gz", "*.zip"])
    plans = plan_key_copies(
        source_keys=[obj.relative_path for obj in source_objects],
        source_bucket=source.bucket,
        dest_bucket=dest_bucket,
        product="core",
        index=index,
    )
    if not plans:
        raise FileNotFoundError(f"No python packages found at {source.s3_uri}")

    for plan in plans:
        logger.info("Python package: %s -> %s", plan.source.s3_uri, plan.dest.s3_uri)
        backend.copy_file(plan.source, plan.dest)
    logger.info("Copied %d python package files (structured)", len(plans))


def publish_native_linux_packages(
    artifacts_root: WorkflowOutputRoot,
    release_type: str,
    backend: StorageBackend,
    build_variant: str = "release",
) -> None:
    """Copy native Linux packages from the artifacts bucket to the release packages bucket.

    The source packages were uploaded by upload_package_repo.py (called from
    multi_arch_build_native_linux_packages.yml) and already include repodata
    (Packages/Release files for deb, repodata/ for rpm).

    dev/nightly example:
        s3://therock-dev-artifacts/12345-linux/packages/deb/
          -> s3://therock-dev-packages/v4/deb/20250101-12345/
        s3://therock-dev-artifacts/12345-linux/packages/rpm/
          -> s3://therock-dev-packages/v4/rpm/20250101-12345/

    prerelease example:
        s3://therock-prerelease-artifacts/12345-linux/packages/deb/
          -> s3://therock-prerelease-packages/v4/packages/deb/
        s3://therock-prerelease-artifacts/12345-linux/packages/rpm/
          -> s3://therock-prerelease-packages/v4/packages/rpm/

    asan example:
        s3://therock-dev-artifacts/12345-linux/packages/deb/
          -> s3://therock-dev-packages/v4/packages-asan/deb/20250101-12345/
        s3://therock-dev-artifacts/12345-linux/packages/rpm/
          -> s3://therock-dev-packages/v4/packages-asan/rpm/20250101-12345/

    Note (prerelease): This is a plain copy — the repodata already present in the
    packages bucket is overwritten with the repodata from this run. If multiple
    prerelease runs upload packages to the same fixed prefix, earlier packages
    will no longer be referenced by the repodata.
    TODO: Implement a proper repodata merge for the prerelease case, similar to
    the merge logic in upload_package_repo.py (regenerate_repo_metadata_from_s3).
    """
    dest_bucket = get_release_bucket_config(release_type, "packages")
    today = datetime.date.today().strftime("%Y%m%d")
    is_asan = build_variant == "asan"

    for pkg_type in ["deb", "rpm"]:
        source = artifacts_root.native_linux_packages(pkg_type)

        # Determine base path (asan vs non-asan, prerelease vs dated)
        # prerelease: v4/packages/{pkg_type} or v4/packages-asan/{pkg_type}
        # non-prerelease: v4/{pkg_type}/{dated} or v4/packages-asan/{pkg_type}/{dated}
        if is_asan:
            base_path = "v4/packages-asan"
        else:
            base_path = "v4/packages" if release_type == "prerelease" else "v4"

        if release_type == "prerelease":
            dest_prefix = f"{base_path}/{pkg_type}"
        else:
            dest_prefix = f"{base_path}/{pkg_type}/{today}-{artifacts_root.run_id}"

        dest = StorageLocation(dest_bucket.name, dest_prefix)
        logger.info(
            "Native %s packages: %s -> %s", pkg_type, source.s3_uri, dest.s3_uri
        )
        count = backend.copy_directory(source, dest)
        logger.info("Copied %d files for %s packages", count, pkg_type)
        if count == 0:
            raise FileNotFoundError(f"No {pkg_type} packages found at {source.s3_uri}")


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
        "--structured",
        action="store_true",
        help="Publish python packages into product-local package directories "
        "(v5/rocm/core/<index>/<package>/) instead of the flat v4/whl prefix. "
        "Multi-arch (--kpack-split true) only.",
    )
    parser.add_argument(
        "--python-index",
        default="whl",
        choices=["whl", "whl-next"],
        help="Product-local index name for structured Python publishing "
        "(default: whl). Selects the v5/rocm/core/<index>/ path segment.",
    )
    parser.add_argument(
        "--skip-native-packages",
        action="store_true",
        help="Skip publishing native Linux packages (deb/rpm)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan without copying"
    )
    parser.add_argument(
        "--build-variant",
        default="release",
        choices=["release", "asan"],
        help="Build variant (default: release). ASAN builds skip python packages "
        "and publish native packages to separate paths.",
    )
    args = parser.parse_args(argv)

    kpack_split = args.kpack_split.lower() == "true"
    if args.structured and not kpack_split:
        parser.error("--structured requires --kpack-split true (multi-arch only)")

    artifacts_root = WorkflowOutputRoot.from_workflow_run(
        run_id=args.run_id, platform=args.platform, release_type=args.release_type
    )
    backend = create_storage_backend(dry_run=args.dry_run)
    is_asan = args.build_variant == "asan"

    publish_tarballs(artifacts_root, args.release_type, backend, args.build_variant)
    if not is_asan:
        publish_python_packages(
            artifacts_root,
            args.release_type,
            backend,
            kpack_split,
            structured=args.structured,
            index=args.python_index,
        )
    else:
        logger.info("Skipping python packages for ASAN build variant")
    if artifacts_root.platform == "linux" and not args.skip_native_packages:
        publish_native_linux_packages(
            artifacts_root, args.release_type, backend, args.build_variant
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
