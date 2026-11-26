#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
This script uploads built Python packages (wheels, sdists) to S3 or a local
directory for testing. Package indexes are generated server-side. Once uploaded,
packages can be downloaded directly or via `pip install --find-links {url}` by
developers, users, and test workflows.

Usage:
  upload_python_packages.py
    --input-packages-dir PACKAGES_DIR
    --run-id RUN_ID
    [--output-dir OUTPUT_DIR]  # Local output instead of S3
    [--bucket BUCKET]          # Override bucket selection (defaults to auto-select)
    [--dry-run]                # Print what would happen without taking action

Modes:
  1. S3 upload (default): Uploads to an AWS S3 bucket
  2. Local output: With --output-dir, copies files to local directory
  3. Dry run: With --dry-run, prints plan without uploading or copying

Output Layout:
  {bucket}/{external_repo}{run_id}-{platform}/python/
    *.whl, *.tar.gz   # Host and per-target wheels and sdists
    index.html        # Generated server-side for pip --find-links

Installation:
  pip install rocm[libraries,devel] --pre \\
    --find-links=https://{bucket}.s3.amazonaws.com/{path}/index.html
"""

import argparse
from pathlib import Path
import platform
import sys

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))
from _therock_utils.workflow_outputs import WorkflowOutputRoot
from _therock_utils.storage_location import StorageLocation
from _therock_utils.storage_backend import StorageBackend, create_storage_backend
from github_actions_api import (
    gha_append_step_summary,
    gha_set_output,
)

PLATFORM = platform.system().lower()
LINE_CONTINUATION_CHAR = "^" if PLATFORM == "windows" else "\\"


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


def find_package_files(dist_dir: Path) -> list[Path]:
    """Finds all wheel, sdist, and index files in the dist directory."""
    files = []
    for pattern in ["*.whl", "*.tar.gz", "index.html"]:
        files.extend(dist_dir.glob(pattern))

    return sorted(files)


def upload_packages(
    dist_dir: Path, packages_loc: StorageLocation, backend: StorageBackend
):
    """Upload package files using the provided backend."""
    package_files = find_package_files(dist_dir)
    if not package_files:
        raise FileNotFoundError(f"No package files found in {dist_dir}")

    log(f"[INFO] Found {len(package_files)} top-level package files in {dist_dir}:")
    for f in package_files:
        log(f"  - {f.relative_to(dist_dir)}")

    # Log all files that will actually be uploaded (including subdirectories).
    all_files = sorted(
        f for f in dist_dir.rglob("*") if f.is_file() and not f.is_symlink()
    )
    log(f"[INFO] Uploading {len(all_files)} total files to {packages_loc.s3_uri}:")
    for f in all_files:
        log(f"  {f.relative_to(dist_dir).as_posix()}")

    count = backend.upload_directory(dist_dir, packages_loc)
    log(f"[INFO] Uploaded {count} files")


def write_gha_upload_summary(packages_loc: StorageLocation):
    """Write install instructions using one index for all per-target wheels."""
    index_url = f"{packages_loc.https_url}/index.html"
    install_instructions_markdown = f"""[ROCm Python packages (kpack-split)]({index_url})
Replace `<YOUR_TARGET>` with your GPU target (e.g. `gfx942`, `gfx1201`):
```bash
pip install rocm[libraries,devel,device-<YOUR_TARGET>] --pre {LINE_CONTINUATION_CHAR}
    --find-links={index_url}
```
"""
    gha_append_step_summary(install_instructions_markdown)


def run(args: argparse.Namespace):
    packages_dir = args.input_packages_dir.resolve()
    if not packages_dir.is_dir():
        raise FileNotFoundError(f"Packages root directory not found: {packages_dir}")

    # Convention: build_python_packages.py writes into <packages_dir>/dist/,
    # but build_prod_wheels.py (pytorch) writes directly into its output dir.
    # Use the dist/ subdirectory when it exists, otherwise use the input dir.
    dist_subdir = packages_dir / "dist"
    dist_dir = dist_subdir if dist_subdir.is_dir() else packages_dir

    log(f"[INFO] Packages directory  : {packages_dir}")
    log(f"[INFO] Dist [sub]directory : {dist_dir}")
    log(f"[INFO] Run ID              : {args.run_id}")
    log(f"[INFO] Platform            : {PLATFORM}")
    if args.dry_run:
        log(f"[INFO] Mode              : DRY RUN")
    elif args.output_dir:
        log(f"[INFO] Mode              : Local output to {args.output_dir}")
    else:
        log(f"[INFO] Mode              : S3 upload")

    output_root = _make_output_root(args.run_id, bucket_override=args.bucket)
    # ROCm, PyTorch, and JAX jobs upload packages to the same run's python/
    # directory. The server generates a combined index; uploading an index
    # from one job could hide packages uploaded by another.
    packages_loc = output_root.python_packages()
    backend = create_storage_backend(staging_dir=args.output_dir, dry_run=args.dry_run)

    log("")
    log("Uploading packages")
    log("------------------")
    upload_packages(dist_dir=dist_dir, packages_loc=packages_loc, backend=backend)

    if not args.output_dir:
        index_url = f"{packages_loc.https_url}/index.html"
        log(f"[INFO] Package index URL: {index_url}")

        log("Set github actions output")
        log("-------------------------")
        gha_set_output({"package_find_links_url": index_url})

        log("Write github actions build summary")
        log("----------------------------------")
        write_gha_upload_summary(packages_loc)

    log("")
    log("[INFO] Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Upload Python packages to S3 or a local directory"
    )
    parser.add_argument(
        "--input-packages-dir",
        type=Path,
        required=True,
        help="Directory containing built packages (with dist/ subdirectory)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Workflow run ID (e.g. 21440027240)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output to local directory instead of S3",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="Override S3 bucket (default: auto-select from workflow run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without uploading or copying",
    )

    args = parser.parse_args()

    if args.output_dir and args.bucket:
        parser.error("--output-dir and --bucket are mutually exclusive")

    run(args)


if __name__ == "__main__":
    main()
