#!/usr/bin/env python3
"""
upload_logs_to_s3.py

Uploads log files and index.html to an S3 bucket using boto3.
"""

import os
import sys
import argparse
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def upload_file_boto3(file_path: Path, bucket: str, key: str, content_type: str = None):
    s3 = boto3.client("s3")
    extra_args = {"ContentType": content_type} if content_type else {}

    try:
        # Check if file already exists
        s3.head_object(Bucket=bucket, Key=key)
        already_exists = True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            already_exists = False
        else:
            log(f"[ERROR] Failed checking existence for {key}: {e}")
            return

    if already_exists:
        # Upload anyway for the additional logs,
        # but do not repeat success log
        try:
            s3.upload_file(str(file_path), bucket, key, ExtraArgs=extra_args)
        except ClientError as e:
            log(f"[ERROR] Re-upload failed for {file_path}: {e}")
        return  # Suppress success log

    # First-time upload
    try:
        log(f"[INFO] Uploading {file_path} to s3://{bucket}/{key}")
        s3.upload_file(str(file_path), bucket, key, ExtraArgs=extra_args)
    except ClientError as e:
        log(f"[ERROR] Failed to upload {file_path} to s3://{bucket}/{key}: {e}")
    else:
        log(f"[INFO] Successfully uploaded {file_path} to s3://{bucket}/{key}")


def upload_logs_to_s3(s3_base_path: str, build_dir: Path):
    log_dir = build_dir / "logs"

    if not log_dir.is_dir():
        log(f"[INFO] Log directory {log_dir} not found. Skipping upload.")
        return

    try:
        bucket, *prefix_parts = s3_base_path[5:].split("/", 1)
        prefix = prefix_parts[0] if prefix_parts else ""
    except Exception:
        log(f"[ERROR] Could not parse s3_base_path: {s3_base_path}")
        sys.exit(2)

    # Upload .log files
    log_files = list(log_dir.glob("*.log"))
    if not log_files:
        log("[WARN] No .log files found. Skipping log upload.")
    else:
        for file_path in log_files:
            key = f"{prefix}/{file_path.name}" if prefix else file_path.name
            upload_file_boto3(file_path, bucket, key, content_type="text/plain")

    # Upload index.html
    index_path = log_dir / "index.html"
    if index_path.is_file():
        key = f"{prefix}/index.html" if prefix else "index.html"
        upload_file_boto3(index_path, bucket, key, content_type="text/html")
    else:
        log(f"[INFO] No index.html found at {log_dir}. Skipping index upload.")


def main():
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
        "--s3-base-path",
        type=str,
        required=True,
        help="Base S3 path to upload logs to, e.g. s3://bucket/run-id-platform/logs/family",
    )
    args = parser.parse_args()

    upload_logs_to_s3(args.s3_base_path, args.build_dir)


if __name__ == "__main__":
    main()
