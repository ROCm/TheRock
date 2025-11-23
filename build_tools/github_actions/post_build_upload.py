#!/usr/bin/env python3

"""
Usage:
post_build_upload.py [-h]
  --artifact-group ARTIFACT_GROUP
  [--build-dir BUILD_DIR]
  [--upload | --no-upload] (default enabled if the `CI` env var is set)
  [--run-id RUN_ID]

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
from datetime import datetime
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile

from github_actions_utils import *

THEROCK_DIR = Path(__file__).resolve().parent.parent.parent
PLATFORM = platform.system().lower()

# Importing indexer.py
sys.path.append(str(THEROCK_DIR / "third-party" / "indexer"))
import indexer


def log(*args):
    print(*args)
    sys.stdout.flush()


def exec(cmd: list[str], cwd: Path):
    log(f"++ Exec [{cwd}]$ {shlex.join(cmd)}")
    subprocess.run(cmd, check=True)


# This method will output logs of the Windows Time Service and is meant
# to help debug spurious AWS auth issues caused by time differences when
# uploading with the AWS CLI tool. For context, see this issue and PR:
# https://github.com/ROCm/TheRock/issues/875
# https://github.com/ROCm/TheRock/pull/1581#issuecomment-3490177590
def write_time_sync_log():
    if platform.system().lower() != "windows":
        log("[*] Current OS not windows, Skipping.")
        return

    # Logs are from `w32tm` run in Windows HostProcess containers on Azure VMs
    # with `/query /status` and `/stripchart /computer:time.aws.com /dataonly`
    # and are mounted via the readonly H: drive for Github Runner Pods to access
    startfile = Path("H:\\start.log")
    timefile = Path("H:\\time.log")

    # Only output if these files exist in the H: drive as expected on Build VMs
    if startfile.is_file() and timefile.is_file():
        log(f"[*] Checking time sync at: {datetime.now()}")

        log("[*] Start Time Sync Log:")
        log(startfile.read_text())

        log("[*] Time Sync Log (last ~50 lines):")
        timef = open(timefile)
        timelines = timef.readlines()
        log("".join(timelines[-51:]))
    else:
        log("[*] time.log and/or start.log not present in H:")


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


def index_log_files(build_dir: Path, artifact_group: str):
    log_dir = build_dir / "logs"
    index_file = log_dir / "index.html"

    indexer_path = THEROCK_DIR / "third-party" / "indexer" / "indexer.py"

    if log_dir.is_dir():
        log(
            f"[INFO] Found '{log_dir}' directory. Indexing '*.log' and '*.tar.gz' files..."
        )
        exec(
            [
                sys.executable,
                str(indexer_path),
                log_dir.as_posix(),  # unnamed path arg in front of -f
                "-f",
                "*.log",
                "*.tar.gz",  # accepts nargs! Take care not to consume path
            ],
            cwd=Path.cwd(),
        )
    else:
        log(f"[WARN] Log directory '{log_dir}' not found. Skipping indexing.")
        return

    if index_file.exists():
        log(
            f"[INFO] Rewriting links in '{index_file}' with ARTIFACT_GROUP={artifact_group}..."
        )
        content = index_file.read_text()
        updated = content.replace(
            'a href=".."', f'a href="../../index-{artifact_group}.html"'
        )
        index_file.write_text(updated)
        log("[INFO] Log index links updated.")
    else:
        log(f"[WARN] '{index_file}' not found. Skipping link rewrite.")


def index_artifact_files(build_dir: Path):
    artifacts_dir = build_dir / "artifacts"
    log(f"Creating index file at {str(artifacts_dir / 'index.html')}")

    indexer_args = argparse.Namespace()
    indexer_args.filter = ["*.tar.xz*"]
    indexer_args.output_file = "index.html"
    indexer_args.verbose = False
    indexer_args.recursive = False
    indexer.process_dir(artifacts_dir, indexer_args)


def upload_artifacts(artifact_group: str, build_dir: Path, bucket_uri: str):
    log("Uploading artifacts to S3")

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
        "--region",
        "us-east-2",
    ]
    exec(cmd, cwd=Path.cwd())

    # Uploading index.html to S3 bucket
    cmd = [
        "aws",
        "s3",
        "cp",
        str(build_dir / "artifacts" / "index.html"),
        f"{bucket_uri}/index-{artifact_group}.html",
    ]
    exec(cmd, cwd=Path.cwd())


def upload_logs_to_s3(artifact_group: str, build_dir: Path, bucket_uri: str):
    s3_base_path = f"{bucket_uri}/logs/{artifact_group}"
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

    # Upload build_time_analysis.html
    analysis_path = log_dir / "build_time_analysis.html"
    if analysis_path.is_file():
        analysis_s3_dest = f"{s3_base_path}/build_time_analysis.html"
        run_aws_cp(analysis_path, analysis_s3_dest, content_type="text/html")
        log(f"[INFO] Uploaded {analysis_path} to {analysis_s3_dest}")

    # Upload index.html
    index_path = log_dir / "index.html"
    if index_path.is_file():
        index_s3_dest = f"{s3_base_path}/index.html"
        run_aws_cp(index_path, index_s3_dest, content_type="text/html")
        log(f"[INFO] Uploaded {index_path} to {index_s3_dest}")
    else:
        log(f"[INFO] No index.html found at {log_dir}. Skipping index upload.")


def upload_manifest_to_s3(artifact_group: str, build_dir: Path, bucket_uri: str):
    """
    Upload therock_manifest.json to:
      <bucket_uri>/manifests/<artifact_group>/therock_manifest.json
    """

    manifest_path = (
        build_dir / "base" / "aux-overlay" / "build" / "therock_manifest.json"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"therock_manifest.json not found at {manifest_path}")

    dest = f"{bucket_uri}/manifests/{artifact_group}/therock_manifest.json"
    log(f"[INFO] Uploading manifest {manifest_path} -> {dest}")
    run_aws_cp(manifest_path, dest, content_type="application/json")


def write_gha_build_summary(artifact_group: str, bucket_url: str):
    log(f"Adding links to job summary to bucket {bucket_url}")

    log_index_url = f"{bucket_url}/logs/{artifact_group}/index.html"
    gha_append_step_summary(f"[Build Logs]({log_index_url})")

    analysis_url = f"{bucket_url}/logs/{artifact_group}/build_time_analysis.html"
    gha_append_step_summary(f"[Build Time Analysis]({analysis_url})")

    artifact_url = f"{bucket_url}/index-{artifact_group}.html"
    gha_append_step_summary(f"[Artifacts]({artifact_url})")

    manifest_url = f"{bucket_url}/manifests/{artifact_group}/therock_manifest.json"
    gha_append_step_summary(f"[TheRock Manifest]({manifest_url})")


def run(args):
    log("Creating Ninja log archive")
    log("--------------------------")
    create_ninja_log_archive(args.build_dir)

    log(f"Indexing log files in {str(args.build_dir)}")
    log("------------------")
    index_log_files(args.build_dir, args.artifact_group)

    log(f"Indexing artifact files in {str(args.build_dir)}")
    log("------------------")
    index_artifact_files(args.build_dir)

    if not args.upload:
        return

    external_repo_path, bucket = retrieve_bucket_info()
    run_id = args.run_id
    bucket_uri = f"s3://{bucket}/{external_repo_path}{run_id}-{PLATFORM}"
    bucket_url = (
        f"https://{bucket}.s3.amazonaws.com/{external_repo_path}{run_id}-{PLATFORM}"
    )

    log("Write Windows time sync log")
    log("----------------------")
    write_time_sync_log()

    log("Upload build artifacts")
    log("----------------------")
    upload_artifacts(args.artifact_group, args.build_dir, bucket_uri)

    log("Upload log")
    log("----------")
    upload_logs_to_s3(args.artifact_group, args.build_dir, bucket_uri)

    log("Upload manifest")
    log("----------------")
    upload_manifest_to_s3(args.artifact_group, args.build_dir, bucket_uri)

    log("Write github actions build summary")
    log("--------------------")
    write_gha_build_summary(args.artifact_group, bucket_url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post Build Upload steps")
    parser.add_argument(
        "--artifact-group",
        type=str,
        default=os.getenv("ARTIFACT_GROUP"),
        required=True,
        help="Artifact group to upload (default: $ARTIFACT_GROUP)",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(os.getenv("BUILD_DIR", "build")),
        help="Build directory containing logs, artifacts, etc. (default: 'build' or $BUILD_DIR)",
    )
    is_ci = str2bool(os.getenv("CI", "false"))
    parser.add_argument(
        "--upload",
        default=is_ci,
        help="Enable upload steps (default enabled if $CI is set)",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--run-id", type=str, help="GitHub run ID of this workflow run")
    args = parser.parse_args()

    # Check preconditions for provided arguments before proceeding.

    if args.upload:
        if not args.run_id:
            parser.error("when --upload is true, --run_id must also be set")

        if not shutil.which("aws"):
            raise FileNotFoundError(
                "AWS CLI 'aws' not found on PATH, uploading requires it"
            )

    if not args.build_dir.is_dir():
        raise FileNotFoundError(
            f"""
Build directory ({str(args.build_dir)}) not found. Skipping upload!
This can be due to the CI job being cancelled before the build was started.
            """
        )

    run(args)
