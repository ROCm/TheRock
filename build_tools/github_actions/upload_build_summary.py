#!/usr/bin/env python3
"""
upload_job_summary.py

Uploads job summary links to the GitHub step summary
"""

import argparse
from upload_build_artifacts import retrieve_bucket_info
import logging
import os
from pathlib import Path
import platform
import sys

from github_actions_utils import *

logging.basicConfig(level=logging.INFO)

THEROCK_DIR = Path(__file__).resolve().parent.parent.parent
PLATFORM = platform.system().lower()


def run(args: argparse.Namespace):
    external_repo_path, bucket = retrieve_bucket_info()
    run_id = args.run_id
    bucket_url = (
        f"https://{bucket}.s3.amazonaws.com/{external_repo_path}{run_id}-{PLATFORM}"
    )
    logging.info(f"Adding links to job summary to bucket {bucket}")
    build_dir = args.build_dir
    amdgpu_family = args.amdgpu_family

    log_url = f"{bucket_url}/logs/{amdgpu_family}/index.html"
    gha_append_step_summary(f"[Build Logs]({log_url})")
    if os.path.exists(build_dir / "artifacts" / "index.html"):
        artifact_url = f"{bucket_url}/index-{amdgpu_family}.html"
        gha_append_step_summary(f"[Artifacts]({artifact_url})")
    else:
        logging.info("No artifacts index found. Skipping artifact link.")


def main(argv):
    parser = argparse.ArgumentParser(prog="upload_job_summary")
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

    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
