#!/usr/bin/env python3
"""
upload_build_artifacts.py

Uploads build artifacts to AWS S3 bucket
"""

import argparse
import logging
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

logging.basicConfig(level=logging.INFO)

THEROCK_DIR = Path(__file__).resolve().parent.parent.parent
PLATFORM = platform.system().lower()

# Importing indexer.py
sys.path.append(str(THEROCK_DIR / "third-party" / "indexer"))
from indexer import process_dir


def exec(cmd: list[str], cwd: Path):
    logging.info(f"++ Exec [{cwd}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, check=True)


def retrieve_bucket_info() -> tuple[str, str]:
    github_repository = os.getenv("GITHUB_REPOSITORY", "ROCm/TheRock")
    is_pr_from_fork = os.getenv("IS_PR_FROM_FORK", "false") == "true"
    owner, repo_name = github_repository.split("/")
    external_repo = (
        ""
        if repo_name == "TheRock" and owner == "ROCm" and not is_pr_from_fork
        else f"{owner}-{repo_name}/"
    )
    bucket = (
        "therock-artifacts"
        if repo_name == "TheRock" and owner == "ROCm" and not is_pr_from_fork
        else "therock-artifacts-external"
    )
    return (external_repo, bucket)


def create_index_file(args: argparse.Namespace):
    logging.info("Creating index file")
    build_dir = args.build_dir / "artifacts"

    indexer_args = argparse.Namespace()
    indexer_args.filter = ["*.html*"]
    indexer_args.output_file = "index.html"
    indexer_args.verbose = False
    indexer_args.recursive = False
    process_dir(build_dir, indexer_args)


# Enhancement to upload all HTML test reports in a directory
def upload_test_report(report_dir: Path, bucket_uri: str):
    """
    Upload all .html files from report_dir to bucket_uri (keeps filenames).
    """
    if not report_dir.exists() or not report_dir.is_dir():
        logging.error(
            "Report directory %s not found or not a directory — skipping upload.",
            report_dir,
        )
        return

    # Use a single AWS CLI call to copy only *.html files recursively
    cmd = [
        "aws",
        "s3",
        "cp",
        str(report_dir),
        bucket_uri,
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

    # Skip uploading report if report path is not set
    if args.report_path is None:
        logging.error("--report-path is not provided — skipping upload")
    else:
        logging.info(
            "--report-path is set; uploading HTML reports from %s to %s",
            args.report_path,
            bucket_uri,
        )
        upload_test_report(args.report_path, bucket_uri)
        create_index_file(args)


def main(argv):
    parser = argparse.ArgumentParser(prog="artifact_upload")
    parser.add_argument(
        "--run-id", type=str, required=True, help="GitHub run ID of this workflow run"
    )

    parser.add_argument(
        "--amdgpu-family", type=str, required=True, help="AMD GPU family to upload"
    )

    parser.add_argument(
        "--build-dir",
        type=Path,
        required=True,
        help="Path to the build directory of TheRock",
    )

    parser.add_argument(
        "--report-path",
        type=Path,
        help="Directory containing .html files to upload (optional)",
    )

    parser.add_argument(
        "--upload-report",
        action="store_true",  # flips to True if the flag is provided
        default=False,  # default is False when flag is omitted
        help="Knob to Upload test report",
    )

    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
