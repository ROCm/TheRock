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
    if not shutil.which("aws"):
        log("[ERROR] AWS CLI not found in PATH.")
        sys.exit(1)


def run_aws_cp(source_directory: Path, s3_destination: Path, content_type: str = None):
    if source_directory.is_dir():
        cmd = ["aws", "s3", "cp", str(source_directory), str(s3_destination), "--recursive"]
    else:
        cmd = ["aws", "s3", "cp", str(source_directory), str(s3_destination)]

    if content_type:
        cmd += ["--content-type", content_type]
    try:
        log(f"[INFO] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        log(f"[ERROR] Failed to upload {source_directory} to {s3_destination}: {e}")


def upload_logs_to_s3(s3_base_path: str, build_dir: Path):
    log_dir = build_dir / "logs"

    if not log_dir.is_dir():
        log(f"[INFO] Log directory {log_dir} not found. Skipping upload.")
        return

    # Upload .log files
    log_files = list(log_dir.glob("*.log"))
    if not log_files:
        log("[WARN] No .log files found. Skipping log upload.")
    else:
        run_aws_cp(log_dir, s3_base_path, content_type="text/plain")

    # Upload index.html
    index_path = log_dir / "index.html"
    if index_path.is_file():
        index_s3_dest = f"{s3_base_path}/index.html"
        run_aws_cp(index_path, index_s3_dest, content_type="text/html")
        log(f"[INFO] Uploaded {index_path} to {index_s3_dest}")
    else:
        log("[INFO] No index.html found. Skipping index upload.")


def main():
    check_aws_cli_available()

    parser = argparse.ArgumentParser(description="Upload logs to S3.")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build"),
        help="Path to the build directory (default: build)",
    )
    parser.add_argument(
        "--s3-base-path",
        type=str,
        required=True,
        help="Base S3 path to upload logs to, e.g. s3://bucket/run-id-platform/logs/family",
    )
    args = parser.parse_args()

    upload_logs_to_s3(args.s3_base_path, args.build_dir)


if __name__ == "__main__":
    main()
