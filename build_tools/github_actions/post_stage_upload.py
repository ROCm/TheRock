#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Upload logs from a multi-arch CI stage build.

Each multi-arch CI stage job builds a subset of TheRock (e.g., math-libs for
gfx1151). This script archives ninja logs and uploads the stage's log directory
to S3, organized by stage name and (optionally) GPU family:

    {run_id}-{platform}/logs/{stage_name}/                  # generic stages
    {run_id}-{platform}/logs/{stage_name}/{amdgpu_family}/  # per-arch stages

This is the multi-arch counterpart to post_build_upload.py, which handles
single-stage (monolithic) CI builds. Key differences:

- No artifact upload (artifact_manager.py push handles that)
- No manifest upload (deferred to workflow-level, see #1236)
- No index generation (server-side Lambda handles that, see #3331)
- Logs are scoped to one stage, not the entire build

Usage:
    python post_stage_upload.py \\
        --build-dir build \\
        --stage-name math-libs \\
        --amdgpu-family gfx1151 \\
        --run-id ${{ github.run_id }}
"""

import argparse
import os
from pathlib import Path
import platform
import sys
import tarfile

THEROCK_DIR = Path(__file__).resolve().parent.parent.parent

# Add build_tools to path for _therock_utils imports.
sys.path.insert(0, str(THEROCK_DIR / "build_tools"))
from _therock_utils.workflow_outputs import WorkflowOutputRoot
from _therock_utils.storage_backend import create_storage_backend


def log(*args):
    print(*args)
    sys.stdout.flush()


def create_ninja_log_archive(build_dir: Path) -> Path | None:
    """Archive all .ninja_log files from the build directory.

    Returns the archive path, or None if no ninja logs were found.
    """
    log_dir = build_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    found_files = list(build_dir.glob("**/.ninja_log"))
    if not found_files:
        log("[INFO] No .ninja_log files found. Skipping archive.")
        return None

    archive_path = log_dir / "ninja_logs.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for file_path in found_files:
            tar.add(file_path)
            log(f"[+] Archived: {file_path}")

    log(f"[INFO] Created ninja log archive: {archive_path} ({len(found_files)} files)")
    return archive_path


def upload_stage_logs(
    build_dir: Path,
    output_root: WorkflowOutputRoot,
    stage_name: str,
    amdgpu_family: str,
    dry_run: bool = False,
    output_dir: Path | None = None,
):
    """Upload the stage's log directory to S3."""
    log_dir = build_dir / "logs"
    if not log_dir.is_dir():
        log(f"[INFO] Log directory {log_dir} not found. Skipping upload.")
        return

    dest = output_root.stage_log_dir(stage_name, amdgpu_family)
    backend = create_storage_backend(staging_dir=output_dir, dry_run=dry_run)

    log(f"[INFO] Uploading {log_dir} -> {dest.s3_uri}")
    count = backend.upload_directory(log_dir, dest)
    log(f"[INFO] Uploaded {count} log files")


def run(args: argparse.Namespace):
    log(f"Creating ninja log archive for stage '{args.stage_name}'")
    create_ninja_log_archive(args.build_dir)

    output_root = WorkflowOutputRoot.from_workflow_run(
        run_id=args.run_id,
        platform=platform.system().lower(),
    )

    upload_stage_logs(
        build_dir=args.build_dir,
        output_root=output_root,
        stage_name=args.stage_name,
        amdgpu_family=args.amdgpu_family,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
    )


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Upload logs from a multi-arch CI stage build"
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(os.environ.get("BUILD_DIR", "build")),
        help="Build directory containing logs/ (default: $BUILD_DIR or 'build')",
    )
    parser.add_argument(
        "--stage-name",
        type=str,
        required=True,
        help="Stage name (e.g., 'foundation', 'math-libs')",
    )
    parser.add_argument(
        "--amdgpu-family",
        type=str,
        default="",
        help="GPU family for per-arch stages (e.g., 'gfx1151'). "
        "Empty for generic stages.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=os.environ.get("GITHUB_RUN_ID"),
        help="GitHub Actions run ID (default: $GITHUB_RUN_ID)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write to local directory instead of S3 (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without uploading",
    )

    args = parser.parse_args(argv)

    if not args.run_id:
        parser.error("--run-id is required (or set $GITHUB_RUN_ID)")

    if not args.build_dir.is_dir():
        raise FileNotFoundError(
            f"Build directory not found: {args.build_dir}. "
            "This can happen if the CI job was cancelled before the build started."
        )

    run(args)


if __name__ == "__main__":
    main()
