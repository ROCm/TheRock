#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Upload PyTorch manifest JSON files to S3.

Upload layout:
  s3://{bucket}/{run_id}-{platform}/manifests/pytorch/{amdgpu_family}/
"""

import argparse
from pathlib import Path
import platform
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _therock_utils.workflow_outputs import WorkflowOutputRoot
from _therock_utils.storage_backend import create_storage_backend


PLATFORM = platform.system().lower()


def log(*args):
    print(*args)
    sys.stdout.flush()


def _make_output_root(
    run_id: str, bucket_override: str | None = None
) -> WorkflowOutputRoot:
    if bucket_override:
        return WorkflowOutputRoot(
            bucket=bucket_override, external_repo="", run_id=run_id, platform=PLATFORM
        )
    return WorkflowOutputRoot.from_workflow_run(run_id=run_id, platform=PLATFORM)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload PyTorch manifest JSON files to S3."
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help="Local directory containing manifest JSON files.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Workflow run ID (e.g. 21440027240).",
    )
    parser.add_argument(
        "--amdgpu-family",
        type=str,
        default="",
        help="AMDGPU family (e.g. gfx94X-dcgpu). If empty, uploads without family scoping.",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="Override S3 bucket (default: auto-select from workflow run).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output to local directory instead of S3 (for testing).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without actually uploading.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)

    if not args.manifest_dir.is_dir():
        raise FileNotFoundError(f"Manifest directory not found: {args.manifest_dir}")

    output_root = _make_output_root(args.run_id, bucket_override=args.bucket)
    dest = output_root.pytorch_manifest_dir(args.amdgpu_family)

    backend = create_storage_backend(staging_dir=args.output_dir, dry_run=args.dry_run)
    log(f"Uploading manifests: {args.manifest_dir} -> {dest.s3_uri}")
    count = backend.upload_directory(args.manifest_dir, dest, include=["*.json"])
    log(f"Uploaded {count} manifest file(s)")
    if count == 0:
        raise FileNotFoundError(f"No JSON files found in {args.manifest_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
