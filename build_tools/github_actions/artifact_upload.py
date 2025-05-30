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
GENERIC_VARIANT = "generic"
PLATFORM = platform.system().lower()

# Importing indexer.py
sys.path.append(THEROCK_DIR / "third-party" / "indexer")
from indexer import process_dir

# Importing create_log_index.py and upload_logs_to_s3.py
sys.path.append(THEROCK_DIR / "build_tools")
from create_log_index import index_log_files
from upload_logs_to_s3 import upload_logs_to_s3


def exec(cmd):
    logging.info(f"Executing cmd {shlex.join(cmd)}")
    subprocess.run(cmd, check=True)


def retrieve_bucket_info():
    GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY", "ROCm/TheRock")
    OWNER, REPO_NAME = GITHUB_REPOSITORY.split("/")
    EXTERNAL_REPO = (
        "" if REPO_NAME == "TheRock" and OWNER == "ROCm" else f"{OWNER}-{REPO_NAME}/"
    )
    BUCKET = (
        "therock-artifacts"
        if REPO_NAME == "TheRock" and OWNER == "ROCm"
        else "therock-artifacts-external"
    )
    return (EXTERNAL_REPO, BUCKET)


def set_github_step_summary(summary: str):
    logging.info(f"Appending to github summary: {summary}")
    step_summary_file = os.environ.get("GITHUB_STEP_SUMMARY", "")
    with open(step_summary_file, "a") as f:
        f.write(summary + "\n")


def create_index_file(args: argparse.Namespace):
    logging.info("Creating index file")
    build_dir = args.build_dir / "artifacts"

    indexer_args = argparse.Namespace()
    indexer_args.filter = "*.tar.xz*"
    process_dir(build_dir, indexer_args)


def create_log_index(args: argparse.Namespace):
    logging.info("Creating log index file")
    build_dir = args.build_dir
    amdgpu_family = args.amdgpu_family

    index_log_files(build_dir, amdgpu_family)


def upload_artifacts(args: argparse.Namespace, bucket_uri: str):
    logging.info("Uploading artifacts to S3")
    build_dir = args.build_dir
    amdgpu_family = args.amdgpu_family

    # Uploading artifacts to S3 bucket
    cmd = [
        "aws",
        "s3",
        "cp",
        str(build_dir / "artifacts"),
        bucket_uri,
        "--recursive",
        "--no-follow-symlinks",
        "--exclude",
        "*",
        "--include",
        "*.tar.xz*",
    ]
    exec(cmd)

    # Uploading index.html to S3 bucket
    cmd = [
        "aws",
        "s3",
        "cp",
        str(build_dir / "artifacts" / "index.html"),
        f"{bucket_uri}/index-{amdgpu_family}.html",
    ]
    exec(cmd)


def upload_logs(args: argparse.Namespace, bucket: str, bucket_uri: str):
    logging.info(f"Uploading logs to S3 for bucket {bucket}")
    build_dir = args.build_dir
    amdgpu_family = args.amdgpu_family
    s3_base_path = f"{bucket_uri}/logs/{amdgpu_family}"

    upload_logs_to_s3(s3_base_path, build_dir)


def add_links_to_job_summary(args: argparse.Namespace, bucket: str, bucket_url: str):
    logging.info(f"Adding links to job summary to bucket {bucket}")
    build_dir = args.build_dir
    amdgpu_family = args.amdgpu_family

    log_url = f"{bucket_url}/logs/{amdgpu_family}/index.html"
    set_github_step_summary(f"[Build Logs]({log_url})")
    if os.path.exists(build_dir / "artifacts" / "index.html"):
        artifact_url = f"{bucket_url}/index-{amdgpu_family}.html"
        set_github_step_summary(f"[Artifacts]({artifact_url})")
    else:
        logging.info("No artifacts index found. Skipping artifact link.")


def run(args: argparse.Namespace):
    external_repo_path, bucket = retrieve_bucket_info()
    run_id = args.run_id
    bucket_uri = f"s3://{bucket}/{external_repo_path}{run_id}-{PLATFORM}"
    bucket_url = (
        f"https://{bucket}.s3.amazonaws.com/{external_repo_path}{run_id}-{PLATFORM}"
    )

    create_index_file(args)
    create_log_index(args)
    upload_artifacts(args, bucket_uri)
    upload_logs(args, bucket, bucket_uri)
    add_links_to_job_summary(args, bucket, bucket_url)


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

    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
