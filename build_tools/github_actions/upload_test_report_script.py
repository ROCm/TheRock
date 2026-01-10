#!/usr/bin/env python3
"""
Uploads test reports to AWS S3 bucket for a GitHub run ID and AMD GPU family
"""

import argparse
import logging
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from github_actions.github_actions_utils import retrieve_bucket_info


logging.basicConfig(level=logging.INFO)

THEROCK_DIR = Path(__file__).resolve().parent.parent.parent
PLATFORM = platform.system().lower()

# Importing indexer.py
sys.path.append(str(THEROCK_DIR / "third-party" / "indexer"))
from indexer import process_dir


def exec(cmd: list[str], cwd: Path):
    logging.info(f"++ Exec [{cwd}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, check=True)


# Create an index HTML file listing all test reports in report_dir.
# Output file name is args.index_file_name (e.g. "index_rccl_test_report.html").
def create_index_file(args: argparse.Namespace):
    report_dir = args.report_path
    indexer_args = argparse.Namespace()
    indexer_args.filter = ["*.html*"]
    indexer_args.output_file = args.index_file_name
    indexer_args.verbose = False
    indexer_args.recursive = False
    logging.info("Index file to be created: %s", indexer_args.output_file)
    process_dir(report_dir, indexer_args)


def upload_test_report(report_dir: Path, bucket_uri: str, log_destination: str):
    """
    Upload all .html files from report_dir to bucket_uri (keeps filenames).
    """
    if not report_dir.exists() or not report_dir.is_dir():
        logging.error(
            "Report directory %s not found or not a directory — skipping upload.",
            report_dir,
        )
        return

    # Join S3 bucket and log path cleanly by trimming slashes to avoid double “//”.
    # Example: "s3://bucket//logs/" → "s3://bucket/logs/"
    # Resulting upload path:
    # s3://therock-ci-artifacts-external/ROCm-rccl/18718690315-linux/logs/gfx950-dcgpu/index_rccl_test_report.html
    dest_uri = f"{bucket_uri.rstrip('/')}/{log_destination.lstrip('/')}"
    logging.info(
        "Uploading HTML reports from %s to %s",
        report_dir,
        dest_uri,
    )
    # Use a single AWS CLI call to copy only *.html files recursively
    cmd = [
        "aws",
        "s3",
        "cp",
        str(report_dir),
        dest_uri,
        "--recursive",
        "--exclude",
        "*",
        "--include",
        "*.html",
        "--content-type",
        "text/html",
    ]
    exec(cmd, cwd=Path.cwd())
    logging.info("Uploaded all .html files from %s to %s", report_dir, bucket_uri)


def run(args: argparse.Namespace):
    external_repo_path, bucket = retrieve_bucket_info()
    run_id = args.run_id
    bucket_uri = f"s3://{bucket}/{external_repo_path}{run_id}-{PLATFORM}"

    if not args.report_path.exists():
        logging.error(
            "--report-path %s does not exist — skipping upload", args.report_path
        )
        return

    create_index_file(args)
    upload_test_report(args.report_path, bucket_uri, args.log_destination)


def main(argv):
    parser = argparse.ArgumentParser(prog="upload_test_report")
    parser.add_argument(
        "--run-id", type=str, required=True, help="GitHub run ID of this workflow run"
    )

    parser.add_argument(
        "--amdgpu-family", type=str, required=True, help="AMD GPU family to upload"
    )

    parser.add_argument(
        "--report-path",
        type=Path,
        required=True,
        help="Directory containing .html files to upload (optional)",
    )

    parser.add_argument(
        "--log-destination",
        type=str,
        required=True,
        help="Subdirectory in S3 to upload reports",
    )

    parser.add_argument(
        "--index-file-name",
        type=str,
        required=True,
        help="index file name used for indexing test reports",
    )

    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
