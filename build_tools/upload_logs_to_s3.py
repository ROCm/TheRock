#!/usr/bin/env python3
"""
upload_logs_to_s3.py

Uploads log files and index.html to an S3 bucket using the AWS CLI.
"""

import os
import sys
import glob
import shutil
import argparse
import subprocess
from pathlib import Path


def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def check_aws_cli_available():
    if shutil.which("aws") is None:
        log("[ERROR] AWS CLI not found in PATH.")
        sys.exit(1)


def run_aws_cp(src: str, dest: str, content_type: str = None):
    cmd = (
        ["aws", "s3", "cp", src, dest, "--recursive"]
        if os.path.isdir(src)
        else ["aws", "s3", "cp", src, dest]
    )
    if content_type:
        cmd += ["--content-type", content_type]
    try:
        log(f"[INFO] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        log(f"[ERROR] Failed to upload {src} to {dest}: {e}")


def upload_logs_to_s3(
    bucket_name: str, run_id: str, amdgpu_family: str, build_dir: Path
):
    log_dir = build_dir / "logs"
    s3_base_path = f"s3://{bucket_name}/{run_id}-linux/logs/{amdgpu_family}"

    if not log_dir.is_dir():
        log(f"[INFO] Log directory {log_dir} not found. Skipping upload.")
        return

    # Upload .log files
    log_files = list(log_dir.glob("*.log"))
    if not log_files:
        log("[WARN] No .log files found. Skipping log upload.")
    else:
        run_aws_cp(str(log_dir), s3_base_path, content_type="text/plain")

    # Upload index.html
    index_path = log_dir / "index.html"
    if index_path.is_file():
        index_s3_dest = f"{s3_base_path}/index.html"
        run_aws_cp(str(index_path), index_s3_dest, content_type="text/html")
        log(f"[INFO] Uploaded {index_path} to {index_s3_dest}")
    else:
        log("[INFO] No index.html found. Skipping index upload.")


def main():
    check_aws_cli_available()

    # Resolve default directories
    this_script_dir = Path(__file__).resolve().parent
    therock_dir = this_script_dir.parent

    parser = argparse.ArgumentParser(description="Upload logs to S3.")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(os.getenv("BUILD_DIR", therock_dir / "build")),
        help="Path to the build directory (default: repo_root/build or $BUILD_DIR)",
    )
    args = parser.parse_args()

    bucket = os.getenv("S3_BUCKET", "therock-artifacts")
    run_id = os.getenv("GITHUB_RUN_ID")
    amdgpu_family = os.getenv("AMDGPU_FAMILIES")

    if not run_id:
        log("[ERROR] GITHUB_RUN_ID is required.")
        sys.exit(1)
    if not amdgpu_family:
        log("[ERROR] AMDGPU_FAMILIES is required.")
        sys.exit(1)

    upload_logs_to_s3(bucket, run_id, amdgpu_family, args.build_dir)


if __name__ == "__main__":
    main()
