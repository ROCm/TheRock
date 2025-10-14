#!/usr/bin/env python3

"""
Usage:
post_build_upload.py [-h] [--build-dir BUILD_DIR] --amdgpu-family AMDGPU_FAMILY [--run-id RUN_ID] [--upload | --no-upload]

This script runs after building TheRock, where this script does:
1. Create log archives
2. Create log index files
3. (optional) upload artifacts
4. (optional) upload logs
5. (optional) add links to GitHub job summary

In the case that a CI build fails, this step will always upload available logs and artifacts.

For AWS credentials to upload, reach out to the #rocm-ci channel in the AMD Developer Community Discord
"""

import argparse
import os
import tarfile
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys

THEROCK_DIR = Path(__file__).resolve().parent.parent.parent
PLATFORM = platform.system().lower()

# Importing indexer.py
sys.path.append(str(THEROCK_DIR / "third-party" / "indexer"))
from indexer import process_dir
from github_actions_utils import *


def log(*args):
    print(*args)
    sys.stdout.flush()


def exec(cmd: list[str], cwd: Path):
    log(f"++ Exec [{cwd}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, check=True)


def is_windows():
    return platform.system().lower() == "windows"


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


def create_ninja_log_archive(build_dir: Path):
    log_dir = build_dir / "logs"

    # Python equivalent of `find  ~/TheRock/build -iname .ninja_log``
    found_files = []
    log(f"[*] Create ninja log archive from: {build_dir}")

    glob_pattern_ninja = f"**/.ninja_log"
    log(f"[*] Path glob: {glob_pattern_ninja}")
    found_files = list(build_dir.glob(glob_pattern_ninja))

    if len(found_files) == 0:
        print("No ninja log files found to archive... Skipping", file=sys.stderr)
        return

    files_to_archive = found_files
    archive_name = log_dir / "ninja_logs.tar.gz"
    if archive_name.exists():
        print(f"NOTE: Archive exists: {archive_name}", file=sys.stderr)
    added_count = 0
    with tarfile.open(archive_name, "w:gz") as tar:
        log(f"[+] Create archive: {archive_name}")
        for file_path in files_to_archive:
            tar.add(file_path)
            added_count += 1
            log(f"[+]  Add: {file_path}")
    log(f"[*] Files Added: {added_count}")


def index_log_files(build_dir: Path, amdgpu_family: str):
    log_dir = build_dir / "logs"
    index_file = log_dir / "index.html"

    indexer_path = THEROCK_DIR / "third-party" / "indexer" / "indexer.py"

    if log_dir.is_dir():
        log(
            f"[INFO] Found '{log_dir}' directory. Indexing '*.log' and '*.tar.gz' files..."
        )
        subprocess.run(
            [
                "python",
                str(indexer_path),
                log_dir.as_posix(),  # unnamed path arg in front of -f
                "-f",
                "*.log",
                "*.tar.gz",  # accepts nargs! Take care not to consume path
            ],
            check=True,
        )
    else:
        log(f"[WARN] Log directory '{log_dir}' not found. Skipping indexing.")
        return

    if index_file.exists():
        log(
            f"[INFO] Rewriting links in '{index_file}' with AMDGPU_FAMILIES={amdgpu_family}..."
        )
        content = index_file.read_text()
        updated = content.replace(
            'a href=".."', f'a href="../../index-{amdgpu_family}.html"'
        )
        index_file.write_text(updated)
        log("[INFO] Log index links updated.")
    else:
        log(f"[WARN] '{index_file}' not found. Skipping link rewrite.")


def create_index_file(args: argparse.Namespace):
    build_dir = args.build_dir / "artifacts"
    log(f"Creating index file at {str(build_dir / 'index.html')}")

    indexer_args = argparse.Namespace()
    indexer_args.filter = ["*.tar.xz*"]
    indexer_args.output_file = "index.html"
    indexer_args.verbose = False
    indexer_args.recursive = False
    process_dir(build_dir, indexer_args)


def upload_artifacts(args: argparse.Namespace, bucket_uri: str):
    log("Uploading artifacts to S3")
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
    exec(cmd, cwd=Path.cwd())

    # Uploading index.html to S3 bucket
    cmd = [
        "aws",
        "s3",
        "cp",
        str(build_dir / "artifacts" / "index.html"),
        f"{bucket_uri}/index-{amdgpu_family}.html",
    ]
    exec(cmd, cwd=Path.cwd())


def upload_logs_to_s3(run_id: str, amdgpu_family: str, build_dir: Path):
    external_repo_path, bucket = retrieve_bucket_info()
    bucket_uri = f"s3://{bucket}/{external_repo_path}{run_id}-{PLATFORM}"
    s3_base_path = f"{bucket_uri}/logs/{amdgpu_family}"

    log_dir = build_dir / "logs"

    if not log_dir.is_dir():
        log(f"[INFO] Log directory {log_dir} not found. Skipping upload.")
        return

    # Upload .log files
    log_files = list(log_dir.glob("*.log")) + list(log_dir.glob("*.tar.gz"))
    if not log_files:
        log("[WARN] No .log or .tar.gz files found. Skipping log upload.")
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


def upload_build_summary(args):
    external_repo_path, bucket = retrieve_bucket_info()
    run_id = args.run_id
    bucket_url = (
        f"https://{bucket}.s3.amazonaws.com/{external_repo_path}{run_id}-{PLATFORM}"
    )
    log(f"Adding links to job summary to bucket {bucket}")
    build_dir = args.build_dir
    amdgpu_family = args.amdgpu_family

    log_url = f"{bucket_url}/logs/{amdgpu_family}/index.html"
    gha_append_step_summary(f"[Build Logs]({log_url})")
    if os.path.exists(build_dir / "artifacts" / "index.html"):
        artifact_url = f"{bucket_url}/index-{amdgpu_family}.html"
        gha_append_step_summary(f"[Artifacts]({artifact_url})")
    else:
        log("No artifacts index found. Skipping artifact link.")


def run(args):
    if not args.build_dir.is_dir():
        log(
            f"""
[ERROR] No build directory ({str(args.build_dir)}) found. Skipping upload of log files!
        This can be due to the CI job being cancelled before the build was started.
            """
        )
        sys.exit(1)

    log("Creating Ninja log archive")
    log("--------------------------")
    create_ninja_log_archive(args.build_dir)

    log(f"Indexing log files in {str(args.build_dir)}")
    log("------------------")
    index_log_files(args.build_dir, args.amdgpu_family)

    if args.upload:
        check_aws_cli_available()
        log("Upload build artifacts")
        log("----------------------")
        external_repo_path, bucket = retrieve_bucket_info()
        run_id = args.run_id
        bucket_uri = f"s3://{bucket}/{external_repo_path}{run_id}-{PLATFORM}"

        create_index_file(args)
        upload_artifacts(args, bucket_uri)

        log("Upload log")
        log("----------")
        upload_logs_to_s3(args.run_id, args.amdgpu_family, args.build_dir)

        log("Upload build summary")
        log("--------------------")
        upload_build_summary(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post Build Upload steps")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(os.getenv("BUILD_DIR", "build")),
        help="Build directory containing logs (default: 'build' or $BUILD_DIR)",
    )
    parser.add_argument(
        "--amdgpu-family",
        type=str,
        default=os.getenv("AMDGPU_FAMILIES"),
        required=True,
        help="AMDGPU family name (default: $AMDGPU_FAMILIES)",
    )
    parser.add_argument("--run-id", type=str, help="GitHub run ID of this workflow run")
    is_ci = str2bool(os.getenv("CI", "false"))
    parser.add_argument(
        "--upload",
        default=is_ci,
        help="Enable upload steps",
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()
    run(args)
