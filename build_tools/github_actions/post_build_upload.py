#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


"""
Usage:
post_build_upload.py [-h]
  --artifact-group ARTIFACT_GROUP
  [--build-dir BUILD_DIR]
  [--upload | --no-upload] (default enabled if the `CI` env var is set)
  [--run-id RUN_ID]
  [--output-dir OUTPUT_DIR]
  [--dry-run]

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
import subprocess
import sys
import tarfile

from github_actions_utils import gha_append_step_summary, str2bool

THEROCK_DIR = Path(__file__).resolve().parent.parent.parent
PLATFORM = platform.system().lower()

# Add build_tools to path for _therock_utils imports.
sys.path.insert(0, str(THEROCK_DIR / "build_tools"))
from _therock_utils.workflow_outputs import WorkflowOutputRoot
from _therock_utils.storage_backend import StorageBackend, create_storage_backend

# Importing indexer.py
sys.path.append(str(THEROCK_DIR / "third-party" / "indexer"))
import indexer


def log(*args):
    print(*args)
    sys.stdout.flush()


def run_command(cmd: list[str], cwd: Path):
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
        run_command(
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


def upload_artifacts(
    artifact_group: str,
    build_dir: Path,
    output_root: WorkflowOutputRoot,
    backend: StorageBackend,
):
    """Upload build artifacts (.tar.xz archives and checksums) and index."""
    artifacts_dir = build_dir / "artifacts"
    if not artifacts_dir.is_dir():
        log(f"[INFO] Artifacts directory {artifacts_dir} not found. Skipping.")
        return

    log("Uploading artifacts")
    count = backend.upload_directory(
        artifacts_dir, output_root.root(), include=["*.tar.xz*"]
    )
    log(f"[INFO] Uploaded {count} artifact files")

    # Upload the artifact index (generated by index_artifact_files earlier).
    index_path = artifacts_dir / "index.html"
    if index_path.is_file():
        backend.upload_file(index_path, output_root.artifact_index(artifact_group))


def upload_logs(
    artifact_group: str,
    build_dir: Path,
    output_root: WorkflowOutputRoot,
    backend: StorageBackend,
):
    """Upload build logs, resource profiling summaries, and observability reports."""
    log_dir = build_dir / "logs"
    if not log_dir.is_dir():
        log(f"[INFO] Log directory {log_dir} not found. Skipping upload.")
        return

    # Upload all log files recursively. This covers:
    #   - build.log, configure.log, etc.
    #   - ninja_logs.tar.gz
    #   - build_observability.html (when generated)
    #   - index.html (generated by index_log_files)
    #   - therock-build-prof/ subdirectory (resource profiling)
    # Content-type is inferred per file by the backend.
    log("Uploading logs")
    backend.upload_directory(log_dir, output_root.log_dir(artifact_group))

    # Resource profiling summaries (generated by resource_info.py --finalize)
    # live in a subdirectory but are also expected at the log root for direct
    # linking. Upload a flattened copy to preserve current S3 layout.
    resource_prof_dir = log_dir / "therock-build-prof"
    if resource_prof_dir.is_dir():
        for filename in ["comp-summary.html", "comp-summary.md"]:
            file_path = resource_prof_dir / filename
            if file_path.is_file():
                backend.upload_file(
                    file_path, output_root.log_file(artifact_group, filename)
                )
                log(f"[INFO] Uploaded {file_path} (flattened)")


def upload_manifest(
    artifact_group: str,
    build_dir: Path,
    output_root: WorkflowOutputRoot,
    backend: StorageBackend,
):
    """Upload therock_manifest.json."""
    manifest_path = (
        build_dir / "base" / "aux-overlay" / "build" / "therock_manifest.json"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"therock_manifest.json not found at {manifest_path}")

    log(f"[INFO] Uploading manifest {manifest_path}")
    backend.upload_file(manifest_path, output_root.manifest(artifact_group))


def write_gha_build_summary(
    artifact_group: str,
    build_dir: Path,
    output_root: WorkflowOutputRoot,
    job_status: str,
):
    log(f"Adding links to job summary for {output_root.prefix}")

    log_index_url = output_root.log_index(artifact_group).https_url
    gha_append_step_summary(f"[Build Logs]({log_index_url})")

    observability_path = build_dir / "logs" / "build_observability.html"
    if observability_path.is_file():
        analysis_url = output_root.build_observability(artifact_group).https_url
        gha_append_step_summary(f"[Build Observability]({analysis_url})")
    else:
        log("[INFO] Build Observability: Not generated")

    # Only add artifact links if the job not failed
    if not job_status or job_status == "success":
        artifact_url = output_root.artifact_index(artifact_group).https_url
        gha_append_step_summary(f"[Artifacts]({artifact_url})")

    manifest_url = output_root.manifest(artifact_group).https_url
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

    output_root = WorkflowOutputRoot.from_workflow_run(
        run_id=args.run_id, platform=PLATFORM
    )
    backend = create_storage_backend(staging_dir=args.output_dir, dry_run=args.dry_run)

    log("Write Windows time sync log")
    log("----------------------")
    write_time_sync_log()

    # Upload artifacts only if the job not failed
    if not args.job_status or args.job_status == "success":
        log("Upload build artifacts")
        log("----------------------")
        upload_artifacts(args.artifact_group, args.build_dir, output_root, backend)

    log("Upload log")
    log("----------")
    upload_logs(args.artifact_group, args.build_dir, output_root, backend)

    log("Upload manifest")
    log("----------------")
    upload_manifest(args.artifact_group, args.build_dir, output_root, backend)

    log("Write github actions build summary")
    log("--------------------")
    write_gha_build_summary(
        args.artifact_group, args.build_dir, output_root, args.job_status
    )


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
    parser.add_argument(
        "--job-status", type=str, help="Status of this Job ('success', 'failure')"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output to local directory instead of S3 (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without actually uploading",
    )
    args = parser.parse_args()

    # Check preconditions for provided arguments before proceeding.

    if args.upload:
        if not args.run_id:
            parser.error("when --upload is true, --run_id must also be set")

    if not args.build_dir.is_dir():
        raise FileNotFoundError(
            f"""
Build directory ({str(args.build_dir)}) not found. Skipping upload!
This can be due to the CI job being cancelled before the build was started.
            """
        )

    run(args)
