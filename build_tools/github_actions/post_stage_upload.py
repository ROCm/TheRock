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
        --run-id ${{ github.run_id }} \\
        --stage math-libs \\
        --build-dir build \\
        --amdgpu-family gfx1151
"""

import argparse
import logging
import os
from pathlib import Path
import platform
import sys
import tarfile

logging.basicConfig(level=logging.INFO)

THEROCK_DIR = Path(__file__).resolve().parent.parent.parent

# Add build_tools to path for _therock_utils imports.
sys.path.insert(0, str(THEROCK_DIR / "build_tools"))
from _therock_utils.workflow_outputs import WorkflowOutputRoot
from _therock_utils.storage_backend import StorageBackend, create_storage_backend


def log(*args):
    print(*args)
    sys.stdout.flush()


def create_ninja_log_archive(build_dir: Path):
    """Archive all .ninja_log files from the build directory."""
    found_files = list(build_dir.glob("**/.ninja_log"))
    if not found_files:
        log("[INFO] No .ninja_log files found. Skipping archive.")
        return

    log_dir = build_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    archive_path = log_dir / "ninja_logs.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for file_path in found_files:
            arcname = file_path.relative_to(build_dir)
            tar.add(file_path, arcname=arcname)
            log(f"[+] Archived: {arcname}")

    log(f"[INFO] Created ninja log archive: {archive_path} ({len(found_files)} files)")
    return archive_path


def _get_pyzstd():
    """Lazy import pyzstd with helpful error message."""
    try:
        import pyzstd

        return pyzstd
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "pyzstd is required for zstd compression. "
            "Install it with: pip install pyzstd"
        )


def create_ccache_log_archive(build_dir: Path, compression_level: int | None = None):
    """Archive the ccache log subdirectory into a zstd-compressed tarball.

    ccache.log can be hundreds of MB (verbose per-invocation trace) but
    compresses ~13x with zstd. The raw logs in logs/ccache/ are excluded from
    upload (see upload_stage_logs); this archive provides the compressed version.
    """
    ccache_dir = build_dir / "logs" / "ccache"
    if not ccache_dir.is_dir():
        return

    found_files = sorted(f for f in ccache_dir.iterdir() if f.is_file())
    if not found_files:
        return

    pyzstd = _get_pyzstd()
    level = compression_level if compression_level is not None else 3
    archive_path = build_dir / "logs" / "ccache_logs.tar.zst"
    with pyzstd.ZstdFile(archive_path, mode="wb", level_or_option=level) as zst:
        with tarfile.open(mode="w|", fileobj=zst) as tar:
            for file_path in found_files:
                tar.add(file_path, arcname=file_path.name)
                log(f"[+] Archived ccache log: {file_path.name}")

    archive_size = archive_path.stat().st_size
    log(
        f"[INFO] Created {archive_path.name} "
        f"({archive_size // 1024 // 1024}MB from {len(found_files)} files)"
    )


def upload_stage_logs(
    build_dir: Path,
    output_root: WorkflowOutputRoot,
    backend: StorageBackend,
    stage_name: str,
    amdgpu_family: str,
):
    """Upload the stage's log directory.

    Args:
        build_dir: Build directory containing logs/.
        output_root: Workflow output root for path computation.
        backend: Storage backend (S3 or local) to upload through.
        stage_name: Build stage (e.g., 'foundation', 'math-libs').
        amdgpu_family: GPU family (e.g., 'gfx1151'). Empty for generic stages.
    """
    log_dir = build_dir / "logs"
    if not log_dir.is_dir():
        log(f"[INFO] Log directory {log_dir} not found. Skipping upload.")
        return

    dest = output_root.log_stage_dir(stage_name, amdgpu_family)
    # Exclude raw ccache logs — they're uploaded compressed as ccache_logs.tar.zst.
    backend.upload_directory(log_dir, dest, exclude=["ccache/**/*"])


def run(args: argparse.Namespace):
    log(f"Creating log archives for stage '{args.stage}'")
    create_ninja_log_archive(args.build_dir)
    create_ccache_log_archive(args.build_dir, compression_level=args.compression_level)

    output_root = WorkflowOutputRoot.from_workflow_run(
        run_id=args.run_id,
        platform=platform.system().lower(),
    )
    backend = create_storage_backend(staging_dir=args.output_dir, dry_run=args.dry_run)

    upload_stage_logs(
        build_dir=args.build_dir,
        output_root=output_root,
        backend=backend,
        stage_name=args.stage,
        amdgpu_family=args.amdgpu_family,
    )


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Post build stage upload steps")
    parser.add_argument(
        "--run-id",
        type=str,
        default=os.environ.get("GITHUB_RUN_ID"),
        help="GitHub Actions run ID (default: $GITHUB_RUN_ID)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        help="Build stage name (e.g., 'foundation', 'math-libs')",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(os.environ.get("BUILD_DIR", "build")),
        help="Build directory containing logs, etc. (default: $BUILD_DIR or 'build')",
    )
    parser.add_argument(
        "--amdgpu-family",
        type=str,
        default="",
        help="GPU family for per-arch stages (e.g., 'gfx1151'). "
        "Empty for generic stages.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=None,
        help="Compression level for zstd archives (default: 3)",
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
        log(
            f"[INFO] Build directory not found: {args.build_dir}. "
            "Nothing to upload (job may have been cancelled before building)."
        )
        return

    run(args)


if __name__ == "__main__":
    main()
