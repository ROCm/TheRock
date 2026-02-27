#!/usr/bin/env python3
"""
Uploads test reports to AWS S3 bucket for a GitHub run ID and AMD GPU family
"""

import argparse
import logging
import sys
from pathlib import Path
import platform

import boto3
from botocore.exceptions import ClientError

from github_actions.github_actions_utils import retrieve_bucket_info


logging.basicConfig(level=logging.INFO)

THEROCK_DIR = Path(__file__).resolve().parent.parent.parent
PLATFORM = platform.system().lower()

# Importing indexer.py
sys.path.append(str(THEROCK_DIR / "third-party" / "indexer"))
from indexer import process_dir


def create_index_file(args: argparse.Namespace):
    """
    Create an index HTML file listing all test reports in report_dir.
    """
    report_dir = args.report_path

    indexer_args = argparse.Namespace()
    indexer_args.filter = ["*.html*"]
    indexer_args.output_file = args.index_file_name
    indexer_args.verbose = False
    indexer_args.recursive = False

    logging.info("Index file to be created: %s", indexer_args.output_file)
    process_dir(report_dir, indexer_args)


def parse_s3_uri(bucket_uri: str):
    """
    Parse s3://bucket-name/prefix into (bucket_name, prefix)
    """
    if not bucket_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {bucket_uri}")

    bucket_uri = bucket_uri.replace("s3://", "", 1)
    parts = bucket_uri.split("/", 1)

    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    return bucket_name, prefix


def upload_test_report(report_dir: Path, bucket_uri: str, log_destination: str):
    """
    Upload all .html files from report_dir to bucket_uri (keeps filenames).
    Equivalent to:
        aws s3 cp <dir> <dest> --recursive --exclude "*" --include "*.html"
    """

    if not report_dir.exists() or not report_dir.is_dir():
        logging.error(
            "Report directory %s not found or not a directory — skipping upload.",
            report_dir,
        )
        return

    bucket_name, base_prefix = parse_s3_uri(bucket_uri)

    # Clean destination path to avoid double slashes
    log_destination = log_destination.strip("/")
    base_prefix = base_prefix.strip("/")

    full_prefix = "/".join(filter(None, [base_prefix, log_destination]))

    logging.info(
        "Uploading HTML reports from %s to s3://%s/%s",
        report_dir,
        bucket_name,
        full_prefix,
    )

    s3_client = boto3.client("s3")

    uploaded_count = 0

    # Non-recursive to match your current behavior
    for file_path in report_dir.glob("*.html"):
        if not file_path.is_file():
            continue

        s3_key = "/".join(filter(None, [full_prefix, file_path.name]))

        try:
            logging.info(
                "Uploading %s → s3://%s/%s",
                file_path,
                bucket_name,
                s3_key,
            )

            s3_client.upload_file(
                Filename=str(file_path),
                Bucket=bucket_name,
                Key=s3_key,
                ExtraArgs={
                    "ContentType": "text/html",
                },
            )

            uploaded_count += 1

        except ClientError as e:
            logging.error(
                "Failed to upload %s to s3://%s/%s: %s",
                file_path,
                bucket_name,
                s3_key,
                e,
            )
            raise

    logging.info(
        "Uploaded %d .html files from %s to s3://%s/%s",
        uploaded_count,
        report_dir,
        bucket_name,
        full_prefix,
    )


def run(args: argparse.Namespace):
    external_repo_path, bucket = retrieve_bucket_info()

    run_id = args.run_id
    bucket_uri = f"s3://{bucket}/{external_repo_path}{run_id}-{PLATFORM}"

    if not args.report_path.exists():
        logging.error(
            "--report-path %s does not exist — skipping upload",
            args.report_path,
        )
        return

    create_index_file(args)
    upload_test_report(args.report_path, bucket_uri, args.log_destination)


def main(argv):
    parser = argparse.ArgumentParser(prog="upload_test_report")

    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="GitHub run ID of this workflow run",
    )

    parser.add_argument(
        "--amdgpu-family",
        type=str,
        required=True,
        help="AMD GPU family to upload",
    )

    parser.add_argument(
        "--report-path",
        type=Path,
        required=True,
        help="Directory containing .html files to upload",
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
        help="Index file name used for indexing test reports",
    )

    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
