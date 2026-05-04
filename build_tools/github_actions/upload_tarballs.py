#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Upload tarballs to S3.

Uploads all .tar.gz files from the input directory to the tarballs/
subdirectory of the workflow output root in S3.

Example with ``--run-id 12345 --platform linux --release-type dev``:

    /tmp/tarballs/therock-dist-linux-gfx94X-dcgpu-7.10.0.tar.gz
      -> s3://therock-dev-artifacts/12345-linux/tarballs/therock-dist-linux-gfx94X-dcgpu-7.10.0.tar.gz

Usage:
    python build_tools/github_actions/upload_tarballs.py \\
        --input-tarballs-dir /tmp/tarballs \\
        --run-id 12345 --platform linux --release-type dev

    python build_tools/github_actions/upload_tarballs.py \\
        --input-tarballs-dir /tmp/tarballs \\
        --run-id 12345 --platform linux --release-type dev --dry-run

    # Local testing (no S3 credentials needed):
    python build_tools/github_actions/upload_tarballs.py \\
        --input-tarballs-dir /tmp/tarballs \\
        --run-id 12345 --platform linux \\
        --output-dir /tmp/upload-test
"""

import argparse
import json
import logging
import platform as platform_module
import sys
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from _therock_utils.storage_backend import create_storage_backend
from _therock_utils.workflow_outputs import WorkflowOutputRoot
from github_actions_api import gha_set_output

logger = logging.getLogger(__name__)


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Unexpected S3 URI: {s3_uri}")
    bucket_and_key = s3_uri[len("s3://") :]
    bucket, prefix = bucket_and_key.split("/", 1)
    return bucket, prefix


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Upload tarballs to S3")
    parser.add_argument(
        "--input-tarballs-dir",
        type=Path,
        required=True,
        help="Directory containing .tar.gz tarballs to upload",
    )
    parser.add_argument("--run-id", required=True, help="Workflow run ID")
    parser.add_argument(
        "--platform",
        default=platform_module.system().lower(),
        choices=["linux", "windows"],
        help="Platform (default: current system)",
    )
    parser.add_argument(
        "--release-type",
        default="",
        help='Release type: "" for CI, or "dev", "nightly", "prerelease"',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output to local directory instead of S3 (for testing)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan without uploading"
    )
    args = parser.parse_args(argv)

    tarballs_dir = args.input_tarballs_dir.resolve()
    if not tarballs_dir.is_dir():
        raise FileNotFoundError(f"Tarballs directory not found: {tarballs_dir}")

    tarball_files = sorted(tarballs_dir.glob("*.tar.gz"))
    if not tarball_files:
        raise FileNotFoundError(f"No .tar.gz files found in {tarballs_dir}")

    logger.info("Found %d tarballs in %s:", len(tarball_files), tarballs_dir)
    for f in tarball_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info("  %s (%.1f MB)", f.name, size_mb)

    output_root = WorkflowOutputRoot.from_workflow_run(
        run_id=args.run_id,
        platform=args.platform,
        release_type=args.release_type or None,
    )
    dest = output_root.tarballs()
    logger.info("Destination: %s", dest.s3_uri)

    backend = create_storage_backend(staging_dir=args.output_dir, dry_run=args.dry_run)
    count = backend.upload_directory(tarballs_dir, dest, include=["*.tar.gz"])

    logger.info("Uploaded %d files", count)
    tarball_urls: dict[str, str] = {}
    bucket, prefix = parse_s3_uri(dest.s3_uri)

    for f in tarball_files:
        # Existing tarball naming convention:
        # therock-dist-<platform>-<family>-<package_version>.tar.gz
        name = f.name
        if not name.startswith(f"therock-dist-{args.platform}-") or not name.endswith(
            ".tar.gz"
        ):
            raise ValueError(f"Unexpected tarball name: {name}")

        family_and_version = name[
            len(f"therock-dist-{args.platform}-") : -len(".tar.gz")
        ]
        family = family_and_version.rsplit("-", 1)[0]
        tarball_urls[family] = f"https://{bucket}.s3.amazonaws.com/{prefix}/{name}"

    gha_set_output({"tarball_urls": json.dumps(tarball_urls)})
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main(sys.argv[1:]))
