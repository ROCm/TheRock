#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Uploads test reports to AWS S3 bucket for a GitHub run ID and AMD GPU family or report type.

TODO: Migrate to StorageBackend (like post_build_upload.py) to replace the
raw `aws s3 cp` calls and gain --output-dir / --dry-run support.
"""

import argparse
import logging
from pathlib import Path
import platform
import shlex
import subprocess
import sys

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from _therock_utils.workflow_outputs import WorkflowOutputRoot
from github_actions.github_actions_api import gha_append_step_summary

logging.basicConfig(level=logging.INFO)

THEROCK_DIR = _BUILD_TOOLS_DIR.parent
PLATFORM = platform.system().lower()

# Importing indexer.py
sys.path.append(str(THEROCK_DIR / "third-party" / "indexer"))
from indexer import process_dir


def run_command(cmd: list[str], cwd: Path):
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


def upload_test_report(report_dir: Path, dest_s3_uri: str):
    """
    Upload all .html files from report_dir to dest_s3_uri (keeps filenames).
    """
    if not report_dir.exists() or not report_dir.is_dir():
        logging.error(
            "Report directory %s not found or not a directory — skipping upload.",
            report_dir,
        )
        return

    logging.info(
        "Uploading HTML reports from %s to %s",
        report_dir,
        dest_s3_uri,
    )
    cmd = [
        "aws",
        "s3",
        "cp",
        str(report_dir),
        dest_s3_uri.rstrip("/") + "/",
        "--recursive",
        "--exclude",
        "*",
        "--include",
        "*.html",
        "--content-type",
        "text/html",
    ]
    run_command(cmd, cwd=Path.cwd())
    logging.info("Uploaded all .html files from %s to %s", report_dir, dest_s3_uri)


def run(args: argparse.Namespace):
    output_root = WorkflowOutputRoot.from_workflow_run(
        run_id=args.run_id, platform=PLATFORM
    )

    if not args.report_path.exists():
        logging.error(
            "--report-path %s does not exist — skipping upload", args.report_path
        )
        return

    # Destination: canonical path from WorkflowOutputRoot when --log-destination
    # is omitted or empty; otherwise legacy base_uri + log_destination (backward compat).
    log_dest = (args.log_destination or "").strip()
    if log_dest:
        base_uri = f"s3://{output_root.bucket}/{output_root.prefix}"
        dest_s3_uri = f"{base_uri.rstrip('/')}/{log_dest.lstrip('/')}"
    else:
        dest_s3_uri = output_root.log_dir(args.amdgpu_family).s3_uri

    create_index_file(args)
    upload_test_report(args.report_path, dest_s3_uri)

    report_url = output_root.log_file(
        args.amdgpu_family, args.index_file_name
    ).https_url
    gha_append_step_summary(f"[Report (S3)]({report_url})")


def main(argv):
    parser = argparse.ArgumentParser(prog="upload_test_report")
    parser.add_argument(
        "--run-id", type=str, required=True, help="GitHub run ID of this workflow run"
    )

    parser.add_argument(
        "--amdgpu-family",
        type=str,
        required=True,
        help="AMD GPU family or report/artifact group for log dir (e.g. gfx950-dcgpu or manifest-diff).",
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
        default=None,
        help=(
            "Subdirectory in S3 to upload reports (legacy). If omitted, destination "
            "is derived from WorkflowOutputRoot.log_dir(amdgpu_family) per workflow_outputs."
        ),
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
