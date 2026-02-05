#!/usr/bin/env python3

"""
This script uploads built Python packages (wheels, sdists) along with an index
page to S3 or a local directory for testing. Once packages are uploaded, they
can be downloaded directly or via `pip install --find-links {url}` by
developers, users, and test workflows.

Usage:
  upload_python_packages.py
    --input-packages-dir PACKAGES_DIR
    --artifact-group ARTIFACT_GROUP
    --run-id RUN_ID
    [--output-dir OUTPUT_DIR]  # Local output instead of S3
    [--bucket BUCKET]          # Override bucket selection (defaults to retrieve_bucket_info())
    [--dry-run]                # Print what would happen without taking action

Modes:
  1. S3 upload (default): Uploads to an AWS S3 bucket
  2. Local output: With --output-dir, copies files to local directory
  3. Dry run: With --dry-run, prints plan without uploading or copying

Output Layout:
  {bucket}/{external_repo}{run_id}-{platform}/python/{artifact_group}/
    *.whl, *.tar.gz   # Wheel and sdist files
    index.html        # File listing for pip --find-links

Installation:
  pip install rocm[libraries,devel] --pre \\
    --find-links=https://{bucket}.s3.amazonaws.com/{path}/index.html
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys

from github_actions_utils import (
    gha_append_step_summary,
    retrieve_bucket_info,
)

THEROCK_DIR = Path(__file__).resolve().parent.parent.parent
PLATFORM = platform.system().lower()
LINE_CONTINUATION_CHAR = "^" if PLATFORM == "windows" else "\\"


def log(*args):
    print(*args)
    sys.stdout.flush()


def run_command(cmd: list[str], cwd: Path = Path.cwd()):
    log(f"++ Exec [{cwd}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, check=True)


# TODO: centralize path construction within "run outputs"
# TODO: document the structure (artifacts, logs, packages, etc.)
@dataclass
class UploadPath:
    """Tracks upload paths and provides S3 URI/URL computation."""

    bucket: str
    prefix: str  # e.g., "21440027240-windows/python/gfx110X-all"

    @property
    def s3_uri(self) -> str:
        """S3 URI for use with aws cli (s3://bucket/prefix)."""
        return f"s3://{self.bucket}/{self.prefix}"

    # TODO: switch to a CDN (cloudfront), downloads direct from S3 are slowww
    @property
    def s3_url(self) -> str:
        """S3 URL for browser/pip access."""
        return f"https://{self.bucket}.s3.amazonaws.com/{self.prefix}"


def build_upload_path_for_workflow_run(
    run_id: str,
    artifact_group: str,
    bucket_override: str | None = None,
) -> UploadPath:
    """Creates an UploadPath for Python package uploads.

    Args:
        run_id: Workflow run ID (e.g., "21440027240")
        artifact_group: Artifact group (e.g., "gfx110X-all")
        bucket_override: Optional bucket name (skips retrieve_bucket_info)

    Returns:
        UploadPath configured for Python packages
    """
    if bucket_override:
        external_repo = ""
        bucket = bucket_override
    else:
        external_repo, bucket = retrieve_bucket_info()

    prefix = f"{external_repo}{run_id}-{PLATFORM}/python/{artifact_group}"
    return UploadPath(bucket=bucket, prefix=prefix)


def generate_index(dist_dir: Path, dry_run: bool = False):
    """Generates an index.html file listing packages for pip --find-links."""
    indexer_script = THEROCK_DIR / "third-party" / "indexer" / "indexer.py"
    if not indexer_script.is_file():
        raise FileNotFoundError(f"Indexer script not found: {indexer_script}")

    cmd = [
        sys.executable,
        str(indexer_script),
        str(dist_dir),
        "--filter",
        "*.whl",
        "*.tar.gz",
    ]

    if dry_run:
        log(f"[DRY RUN] Would run: {shlex.join(cmd)}")
        return

    run_command(cmd)


# TODO: share helper with post_build_upload.py? (that accepts files or dirs)
# TODO: switch to boto3? (just matching existing upload behavior for now)
def run_aws_cp(source_path: Path, s3_destination: str, dry_run: bool = False):
    """Uploads a directory to S3."""
    if not source_path.is_dir():
        raise ValueError(f"source_path must be a directory: {source_path}")

    cmd = ["aws", "s3", "cp", str(source_path), s3_destination, "--recursive"]

    if dry_run:
        log(f"[DRY RUN] Would run: {shlex.join(cmd)}")
        return

    run_command(cmd)


# TODO: share helper with post_build_upload.py?
def run_local_cp(source_path: Path, dest_path: Path, dry_run: bool = False):
    """Copies a directory to a local destination.

    This creates dest_path and its parents as needed.
    """
    if not source_path.is_dir():
        raise ValueError(f"source_path must be a directory: {source_path}")

    if dry_run:
        log(f"[DRY RUN] Would copy {source_path} -> {dest_path}")
        return

    log(f"[INFO] Copying {source_path} -> {dest_path}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        shutil.rmtree(dest_path)
    shutil.copytree(source_path, dest_path)


def find_package_files(dist_dir: Path) -> list[Path]:
    """Finds all wheel, sdist, and index files in the dist directory."""
    files = []
    for pattern in ["*.whl", "*.tar.gz", "index.html"]:
        files.extend(dist_dir.glob(pattern))

    return sorted(files)


def upload_packages(
    dist_dir: Path,
    upload_path: UploadPath,
    output_dir: Path | None = None,
    dry_run: bool = False,
):
    """Uploads package files to S3 or local directory.

    Uploads to a local directory if output_dir is set.
    Otherwise uploads to upload_path.s3_uri.
    """
    package_files = find_package_files(dist_dir)
    if not package_files:
        raise FileNotFoundError(f"No package files found in {dist_dir}")

    log(f"[INFO] Found {len(package_files)} package files in {dist_dir}:")
    for f in package_files:
        log(f"  - {f.relative_to(dist_dir)}")

    # Note: we're not using 'package_files' here, we're just copying/uploading
    # the whole directory. We could check for unexpected/loose files first.

    if output_dir:
        local_dist_path = output_dir / upload_path.prefix
        run_local_cp(
            source_path=dist_dir,
            dest_path=local_dist_path,
            dry_run=dry_run,
        )
    else:
        run_aws_cp(
            source_path=dist_dir,
            s3_destination=upload_path.s3_uri,
            dry_run=dry_run,
        )


def write_gha_upload_summary(upload_path: UploadPath):
    index_url = f"{upload_path.s3_url}/index.html"
    install_instructions_markdown = f"""[ROCm Python packages]({index_url})
```bash
pip install rocm[libraries,devel] --pre {LINE_CONTINUATION_CHAR}
    --find-links={index_url}
```
"""
    gha_append_step_summary(install_instructions_markdown)


def run(args: argparse.Namespace):
    packages_dir = args.input_packages_dir.resolve()
    if not packages_dir.is_dir():
        raise FileNotFoundError(f"Packages root directory not found: {packages_dir}")

    dist_dir = packages_dir / "dist"
    if not dist_dir.is_dir():
        raise FileNotFoundError(f"Packages dist/ subdirectory not found: {dist_dir}")

    log(f"[INFO] Packages directory: {packages_dir}")
    log(f"[INFO] Dist subdirectory : {dist_dir}")
    log(f"[INFO] Artifact group    : {args.artifact_group}")
    log(f"[INFO] Run ID            : {args.run_id}")
    log(f"[INFO] Platform          : {PLATFORM}")
    if args.dry_run:
        log(f"[INFO] Mode              : DRY RUN")
    elif args.output_dir:
        log(f"[INFO] Mode              : Local output to {args.output_dir}")
    else:
        log(f"[INFO] Mode              : S3 upload")

    log("")
    log("Generating index.html")
    log("---------------------")
    generate_index(dist_dir, dry_run=args.dry_run)

    upload_path = build_upload_path_for_workflow_run(
        run_id=args.run_id,
        artifact_group=args.artifact_group,
        bucket_override=args.bucket,
    )

    log("")
    log("Uploading packages")
    log("------------------")
    upload_packages(
        dist_dir=dist_dir,
        upload_path=upload_path,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )

    if not args.output_dir:
        log("Write github actions build summary")
        log("------------------")
        write_gha_upload_summary(upload_path)

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
        "--artifact-group",
        type=str,
        required=True,
        help="Artifact group (e.g., gfx94X-dcgpu)",
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
        help="Override S3 bucket (default: auto-select via retrieve_bucket_info)",
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
