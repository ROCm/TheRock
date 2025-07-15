#!/usr/bin/env python3
"""
upload_logs_to_s3.py

Uploads log files and index.html to an S3 bucket using the AWS CLI.
"""

import argparse
import sys
import shutil
import subprocess
from pathlib import Path
import platform
from upload_build_artifacts import retrieve_bucket_info

PLATFORM = platform.system().lower()


def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def check_aws_cli_available():
    if not shutil.which("aws"):
        log("[ERROR] AWS CLI not found in PATH.")
        sys.exit(1)


def run_aws_cp(source_path: Path, s3_destination: str, content_type: str = None):
    if source_path.is_dir():
        cmd = ["aws", "s3", "cp", str(source_path), s3_destination, "--recursive"]
    else:
        cmd = ["aws", "s3", "cp", str(source_path), s3_destination]

    if content_type:
        cmd += ["--content-type", content_type]
    try:
        log(f"[INFO] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        log(f"[ERROR] Failed to upload {source_path} to {s3_destination}: {e}")


def upload_logs_to_s3(run_id: str, amdgpu_family: str, build_dir: Path):
    external_repo_path, bucket = retrieve_bucket_info()
    bucket_uri = f"s3://{bucket}/{external_repo_path}{run_id}-{PLATFORM}"
    s3_base_path = f"{bucket_uri}/logs/{amdgpu_family}"

    log_dir = build_dir / "logs"

    if not log_dir.is_dir():
        log(f"[INFO] Log directory {log_dir} not found. Skipping upload.")
        return

    # Upload .log files
    log_files = list(log_dir.glob("*"))
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
        log(f"[INFO] No index.html found at {log_dir}. Skipping index upload.")


def main():
    check_aws_cli_available()

    repo_root = Path(__file__).resolve().parent.parent
    default_build_dir = repo_root / "build"

    parser = argparse.ArgumentParser(description="Upload logs to S3.")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=default_build_dir,
        help="Path to the build directory (default: <repo_root>/build)",
    )
    parser.add_argument(
        "--run-id", type=str, required=True, help="GitHub run ID of this workflow run"
    )

    parser.add_argument(
        "--amdgpu-family", type=str, required=True, help="AMD GPU family to upload"
    )
    args = parser.parse_args()

    upload_logs_to_s3(args.run_id, args.amdgpu_family, args.build_dir)


if __name__ == "__main__":
    main()
