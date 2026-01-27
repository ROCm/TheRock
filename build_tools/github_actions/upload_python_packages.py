#!/usr/bin/env python3

"""
Upload Python packages to S3 or a local directory.

Usage:
  upload_python_packages.py
    --packages-dir PACKAGES_DIR
    --artifact-group ARTIFACT_GROUP
    --run-id RUN_ID
    [--output-dir OUTPUT_DIR]  # Local output instead of S3
    [--bucket BUCKET]          # Override bucket selection
    [--dry-run]                # Print what would happen

This script uploads built Python packages (wheels, sdists) to S3 for testing
by downstream workflows. It can also output to a local directory for testing.

Modes:
  1. S3 upload (default): Uploads to S3 bucket selected by retrieve_bucket_info()
  2. Local output: With --output-dir, copies files to local directory
  3. Dry run: With --dry-run, prints plan without uploading or copying

S3 Layout:
  {bucket}/{external_repo}{run_id}-{platform}/python/{artifact_group}/
    dist/           # Wheel and sdist files
    simple/         # Pip index (TODO: not yet implemented)

For AWS credentials, see build_portable_linux_artifacts.yml for the pattern.
"""

import argparse
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys

from github_actions_utils import (
    gha_append_step_summary,
    retrieve_bucket_info,
    str2bool,
)

THEROCK_DIR = Path(__file__).resolve().parent.parent.parent
PLATFORM = platform.system().lower()


def log(*args):
    print(*args)
    sys.stdout.flush()


def run_command(cmd: list[str], cwd: Path | None = None):
    """Run a command and log it."""
    cwd_str = f"[{cwd}]" if cwd else ""
    log(f"++ Exec {cwd_str}$ {shlex.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_aws_cp(
    source_path: Path, s3_destination: str, *, dry_run: bool = False
) -> None:
    """Upload a file or directory to S3."""
    if source_path.is_dir():
        cmd = ["aws", "s3", "cp", str(source_path), s3_destination, "--recursive"]
    else:
        cmd = ["aws", "s3", "cp", str(source_path), s3_destination]

    if dry_run:
        log(f"[DRY RUN] Would run: {shlex.join(cmd)}")
        return

    try:
        log(f"[INFO] Running: {shlex.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        log(f"[ERROR] Failed to upload {source_path} to {s3_destination}: {e}")
        raise


def copy_to_local(source_path: Path, dest_path: Path, *, dry_run: bool = False) -> None:
    """Copy a file or directory to a local destination."""
    if dry_run:
        log(f"[DRY RUN] Would copy {source_path} -> {dest_path}")
        return

    log(f"[INFO] Copying {source_path} -> {dest_path}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if source_path.is_dir():
        if dest_path.exists():
            shutil.rmtree(dest_path)
        shutil.copytree(source_path, dest_path)
    else:
        shutil.copy2(source_path, dest_path)


def find_package_files(packages_dir: Path) -> list[Path]:
    """Find all wheel and sdist files in the packages directory."""
    dist_dir = packages_dir / "dist"
    if not dist_dir.is_dir():
        # Maybe the packages are directly in packages_dir
        dist_dir = packages_dir

    files = []
    for pattern in ["*.whl", "*.tar.gz"]:
        files.extend(dist_dir.glob(pattern))

    return sorted(files)


def compute_paths(
    run_id: str,
    artifact_group: str,
    bucket: str | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Compute all paths for the upload.

    Returns a dict with:
      - bucket: S3 bucket name (or None for local)
      - external_repo: External repo prefix (or empty string)
      - prefix: Full prefix (e.g., "12345-linux/python/gfx94X-dcgpu")
      - dist_path: Path/URI for dist directory
      - simple_path: Path/URI for simple (pip index) directory
      - https_url: HTTPS URL for the upload (for job summary)
    """
    if output_dir:
        # Local output mode
        prefix = f"{run_id}-{PLATFORM}/python/{artifact_group}"
        base_path = output_dir / prefix
        return {
            "bucket": None,
            "external_repo": "",
            "prefix": prefix,
            "dist_path": base_path / "dist",
            "simple_path": base_path / "simple",
            "https_url": None,
        }

    # S3 mode
    if bucket:
        # Override bucket, assume no external_repo prefix
        external_repo = ""
    else:
        external_repo, bucket = retrieve_bucket_info()

    prefix = f"{external_repo}{run_id}-{PLATFORM}/python/{artifact_group}"
    s3_base = f"s3://{bucket}/{prefix}"
    https_base = f"https://{bucket}.s3.amazonaws.com/{prefix}"

    return {
        "bucket": bucket,
        "external_repo": external_repo,
        "prefix": prefix,
        "dist_path": f"{s3_base}/dist/",
        "simple_path": f"{s3_base}/simple/",
        "https_url": https_base,
    }


def upload_packages(
    packages_dir: Path,
    paths: dict,
    *,
    dry_run: bool = False,
) -> None:
    """Upload package files to S3 or local directory."""
    dist_dir = packages_dir / "dist"
    if not dist_dir.is_dir():
        dist_dir = packages_dir

    package_files = find_package_files(packages_dir)
    if not package_files:
        raise FileNotFoundError(f"No package files found in {packages_dir}")

    log(f"[INFO] Found {len(package_files)} package files:")
    for f in package_files:
        log(f"  - {f.name}")

    if paths["bucket"]:
        # S3 upload
        run_aws_cp(dist_dir, paths["dist_path"], dry_run=dry_run)
    else:
        # Local copy
        copy_to_local(dist_dir, paths["dist_path"], dry_run=dry_run)


def write_job_summary(paths: dict, package_files: list[Path]) -> None:
    """Write links to the GitHub Actions job summary."""
    if not paths["https_url"]:
        log("[INFO] No HTTPS URL available, skipping job summary")
        return

    dist_url = f"{paths['https_url']}/dist/"

    summary_lines = [
        "### Python Packages",
        "",
        f"**Packages:** [{dist_url}]({dist_url})",
        "",
        "| Package | Size |",
        "|---------|------|",
    ]

    for f in package_files:
        size_kb = f.stat().st_size / 1024
        if size_kb > 1024:
            size_str = f"{size_kb / 1024:.1f} MB"
        else:
            size_str = f"{size_kb:.1f} KB"
        summary_lines.append(f"| {f.name} | {size_str} |")

    summary = "\n".join(summary_lines)
    log(f"[INFO] Writing job summary:\n{summary}")
    gha_append_step_summary(summary)


def run(args: argparse.Namespace) -> None:
    """Main entry point."""
    packages_dir = args.packages_dir.resolve()
    if not packages_dir.is_dir():
        raise FileNotFoundError(f"Packages directory not found: {packages_dir}")

    # Find package files first to validate input
    package_files = find_package_files(packages_dir)
    if not package_files:
        raise FileNotFoundError(
            f"No package files (*.whl, *.tar.gz) found in {packages_dir}"
        )

    log(f"[INFO] Packages directory: {packages_dir}")
    log(f"[INFO] Artifact group: {args.artifact_group}")
    log(f"[INFO] Run ID: {args.run_id}")
    log(f"[INFO] Platform: {PLATFORM}")
    if args.dry_run:
        log("[INFO] Mode: DRY RUN")
    elif args.output_dir:
        log(f"[INFO] Mode: Local output to {args.output_dir}")
    else:
        log("[INFO] Mode: S3 upload")

    # Compute paths
    paths = compute_paths(
        run_id=args.run_id,
        artifact_group=args.artifact_group,
        bucket=args.bucket,
        output_dir=args.output_dir,
    )

    log(f"[INFO] Destination: {paths['dist_path']}")
    if paths["bucket"]:
        log(f"[INFO] Bucket: {paths['bucket']}")
        if paths["external_repo"]:
            log(f"[INFO] External repo prefix: {paths['external_repo']}")

    # TODO: Generate pip index with piprepo
    # This will create paths["simple_path"] with index.html files

    # Upload/copy packages
    log("")
    log("Uploading packages")
    log("------------------")
    upload_packages(packages_dir, paths, dry_run=args.dry_run)

    # Write job summary (only for S3 uploads, not dry-run)
    if paths["https_url"] and not args.dry_run:
        log("")
        log("Writing job summary")
        log("-------------------")
        write_job_summary(paths, package_files)

    log("")
    log("[INFO] Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Upload Python packages to S3 or local directory"
    )
    parser.add_argument(
        "--packages-dir",
        type=Path,
        required=True,
        help="Directory containing built packages (with dist/ subdirectory)",
    )
    parser.add_argument(
        "--artifact-group",
        type=str,
        required=True,
        help="Artifact group (e.g., gfx94X-dcgpu)",
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
        "--bucket",
        type=str,
        default=None,
        help="Override S3 bucket (default: auto-select via retrieve_bucket_info)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without uploading or copying",
    )

    args = parser.parse_args()

    # Validate conflicting options
    if args.output_dir and args.bucket:
        parser.error("--output-dir and --bucket are mutually exclusive")

    run(args)


if __name__ == "__main__":
    main()
