#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Upload tarballs to S3.

Uploads all .tar.gz files from the input directory to the tarballs/
subdirectory of the workflow output root in S3.

Usage:
    upload_tarballs.py --input-tarballs-dir TARBALLS_DIR --run-id RUN_ID [--platform PLATFORM]

Manual testing:
    # Test with local output (no S3 credentials needed):
    python build_tools/github_actions/upload_tarballs.py \\
        --input-tarballs-dir /tmp/tarballs \\
        --run-id 12345 \\
        --platform linux \\
        --output-dir /tmp/upload-test

    # Verify: /tmp/upload-test/12345-linux/tarballs/*.tar.gz

    # Dry run (prints plan without uploading):
    python build_tools/github_actions/upload_tarballs.py \\
        --input-tarballs-dir /tmp/tarballs \\
        --run-id 12345 \\
        --dry-run
"""

import argparse
import platform as platform_module
import sys
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from _therock_utils.storage_backend import create_storage_backend
from _therock_utils.workflow_outputs import WorkflowOutputRoot


def log(*args):
    print(*args)
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Upload tarballs to S3")
    parser.add_argument(
        "--input-tarballs-dir",
        type=Path,
        required=True,
        help="Directory containing .tar.gz tarballs to upload",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Workflow run ID",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output to local directory instead of S3",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=platform_module.system().lower(),
        choices=["linux", "windows"],
        help="Platform for the upload path (default: current system)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without uploading",
    )
    args = parser.parse_args()

    tarballs_dir = args.input_tarballs_dir.resolve()
    if not tarballs_dir.is_dir():
        raise FileNotFoundError(f"Tarballs directory not found: {tarballs_dir}")

    tarball_files = sorted(tarballs_dir.glob("*.tar.gz"))
    if not tarball_files:
        raise FileNotFoundError(f"No .tar.gz files found in {tarballs_dir}")

    log(f"[INFO] Tarballs directory: {tarballs_dir}")
    log(f"[INFO] Run ID: {args.run_id}")
    log(f"[INFO] Platform: {args.platform}")
    log(f"[INFO] Found {len(tarball_files)} tarballs:")
    for f in tarball_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        log(f"  {f.name} ({size_mb:.1f} MB)")

    output_root = WorkflowOutputRoot.from_workflow_run(
        run_id=args.run_id, platform=args.platform
    )
    tarballs_loc = output_root.tarballs()
    backend = create_storage_backend(staging_dir=args.output_dir, dry_run=args.dry_run)

    log(f"\n[INFO] Uploading to {tarballs_loc.s3_uri}")
    count = backend.upload_directory(tarballs_dir, tarballs_loc, include=["*.tar.gz"])
    log(f"[INFO] Uploaded {count} files")
    log("[INFO] Done!")


if __name__ == "__main__":
    main()
