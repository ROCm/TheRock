#!/usr/bin/env python3
"""
upload_logs_to_s3.py

Uploads log files and index.html to an S3 bucket using boto3.
"""

import os
import sys
import argparse
from pathlib import Path
import time

import boto3
from botocore.exceptions import ClientError


def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def upload_file_boto3(file_path: Path, bucket: str, key: str, content_type: str = None):
    s3 = boto3.client("s3")
    extra_args = {"ContentType": content_type} if content_type else {}

    for attempt in range(3):
        if not file_path.exists() or file_path.stat().st_size == 0:
            log(f"[WARN] Attempt {attempt+1}: index.html is missing or empty")
            time.sleep(1)
        else:
            break
    else:
        log(f"[ERROR] index.html still missing or empty after retries: {file_path}")
        return

    try:
        log(f"[INFO] Uploading {file_path} to s3://{bucket}/{key}")
        s3.upload_file(str(file_path), bucket, key, ExtraArgs=extra_args)
    except ClientError as e:
        log(f"[ERROR] Failed to upload {file_path} to s3://{bucket}/{key}: {e}")
    else:
        log(f"[INFO] Successfully uploaded {file_path} to s3://{bucket}/{key}")


def upload_logs_to_s3(bucket: str, subdir: str, log_dir: Path):

    if not log_dir.is_dir():
        log(f"[INFO] Log directory {log_dir} not found. Skipping upload.")
        return

    # Upload .log files
    log_files = list(log_dir.glob("*.log"))
    if not log_files:
        log("[WARN] No .log files found. Skipping log upload.")
    else:
        for file_path in log_files:
            key = f"{subdir}/{file_path.name}"
            upload_file_boto3(file_path, bucket, key, content_type="text/plain")

    # Upload index.html
    index_path = log_dir / "index.html"
    if index_path.is_file():
        key = f"{subdir}/index.html"
        upload_file_boto3(index_path, bucket, key, content_type="text/html")
    else:
        log(f"[INFO] No index.html found at {log_dir}. Skipping index upload.")


def main():
    default = Path(os.getenv("LOG_DIR", "build/logs"))

    parser = argparse.ArgumentParser(description="Upload logs to S3.")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--subdir", required=True, help="Subdirectory in the bucket")
    parser.add_argument("--log-dir", type=Path, default=default)

    args = parser.parse_args()
    upload_logs_to_s3(args.bucket, args.subdir, args.log_dir)


if __name__ == "__main__":
    main()
