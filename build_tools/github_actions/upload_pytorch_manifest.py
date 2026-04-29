#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Upload the generated PyTorch manifest JSON to S3.

Upload layout:
  s3://{bucket}/{external_repo}{run_id}-{platform}/manifests/{amdgpu_family}/{manifest_name}
"""

import argparse
from pathlib import Path
import platform
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _therock_utils.workflow_outputs import WorkflowOutputRoot
from _therock_utils.storage_location import StorageLocation
from _therock_utils.storage_backend import create_storage_backend


PLATFORM = platform.system().lower()


def log(*args):
    print(*args)
    sys.stdout.flush()


def normalize_python_version_for_filename(python_version: str) -> str:
    """Normalize python version strings for filenames.

    Examples:
      "py3.11" -> "3.11"
      "3.11"   -> "3.11"
    """
    py = python_version.strip()
    if py.startswith("py"):
        py = py[2:]
    return py


def sanitize_ref_for_filename(pytorch_git_ref: str) -> str:
    """Sanitize a git ref for filenames by replacing '/' with '-'.

    Examples:
      "nightly"                -> "nightly"
      "release/2.7"            -> "release-2.7"
      "users/alice/experiment" -> "users-alice-experiment"
    """
    return pytorch_git_ref.replace("/", "-")


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
        description="Upload a PyTorch manifest JSON to S3."
    )
    parser.add_argument(
        "--dist-dir",
        type=Path,
        required=True,
        help="Wheel dist dir (contains manifests/).",
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
        required=True,
        help="AMDGPU family (e.g. gfx94X-dcgpu).",
    )
    parser.add_argument(
        "--python-version",
        type=str,
        required=True,
        help="Python version (e.g. 3.11 or py3.11).",
    )
    parser.add_argument(
        "--pytorch-git-ref",
        type=str,
        required=True,
        help="PyTorch ref (e.g. nightly, release/2.8, users/name/branch).",
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

    py = normalize_python_version_for_filename(args.python_version)
    track = sanitize_ref_for_filename(args.pytorch_git_ref)

    manifest_name = f"therock-manifest_torch_py{py}_{track}.json"
    manifest_path = (args.dist_dir / "manifests" / manifest_name).resolve()

    log(f"Manifest expected at: {manifest_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    output_root = _make_output_root(args.run_id, bucket_override=args.bucket)
    manifest_dir_loc = output_root.manifest_dir(args.amdgpu_family)
    dest = StorageLocation(
        manifest_dir_loc.bucket,
        f"{manifest_dir_loc.relative_path}/{manifest_name}",
    )

    backend = create_storage_backend(staging_dir=args.output_dir, dry_run=args.dry_run)
    backend.upload_file(manifest_path, dest)


if __name__ == "__main__":
    main(sys.argv[1:])
