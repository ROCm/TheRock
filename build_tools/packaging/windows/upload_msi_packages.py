#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Upload built Windows MSI installer packages to the artifacts S3 bucket.

The destination bucket is auto-selected from the RELEASE_TYPE environment
variable (e.g. RELEASE_TYPE=dev -> therock-dev-artifacts), matching how the
Linux native packages and Python packages are uploaded.

Usage:
  upload_msi_packages.py --run-id RUN_ID --package-dir DIR [--dry-run]

Output layout:
  {bucket}/{external_repo}{run_id}-windows/packages/msi/
    *.msi

The published MSIs are later copied into a release bucket by
build_tools/github_actions/publish_rocm_to_release_buckets.py.
"""

import argparse
import sys
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent.parent
if str(_BUILD_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from _therock_utils.storage_backend import create_storage_backend
from _therock_utils.workflow_outputs import WorkflowOutputRoot

# MSIs are Windows installers regardless of the runner this script executes on,
# so the artifact prefix is always the "-windows" one.
PLATFORM = "windows"


def upload_msi_packages(run_id: str, package_dir: Path, dry_run: bool) -> None:
    if not package_dir.is_dir():
        sys.exit(f"Error: package dir not found: {package_dir}")

    output_root = WorkflowOutputRoot.from_workflow_run(run_id=run_id, platform=PLATFORM)
    dest = output_root.native_windows_packages("msi")

    backend = create_storage_backend(dry_run=dry_run)
    print(f"Uploading *.msi from {package_dir} to {dest.s3_uri}")
    count = backend.upload_directory(package_dir, dest, include=["*.msi"])
    print(f"Uploaded {count} MSI file(s)")
    if count == 0:
        raise FileNotFoundError(f"No .msi files found in {package_dir}")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Upload Windows MSI packages to the artifacts bucket."
    )
    parser.add_argument("--run-id", required=True, help="GitHub Actions run ID")
    parser.add_argument(
        "--package-dir",
        required=True,
        type=Path,
        help="Local directory containing the built .msi files",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan without uploading"
    )
    args = parser.parse_args(argv)
    upload_msi_packages(args.run_id, args.package_dir, args.dry_run)


if __name__ == "__main__":
    main(sys.argv[1:])
