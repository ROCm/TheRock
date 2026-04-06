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
  [--artifacts-bucket BUCKET]
  [--path-prefix PREFIX]
  [--summary-base-url URL] [--summary-path PREFIX]
  [--skip-manifest]

Path/URL arguments (--path-prefix, --summary-base-url, --summary-path):
  --path-prefix is the S3 key prefix for all outputs; --summary-path is the
  corresponding prefix in the CDN URL used for job summary links. These let
  private workflows place artifacts under a versioned subdirectory while
  surfacing links through a CDN that maps a different path prefix.

  Example:
    --path-prefix v3/artifacts
    --summary-base-url https://artifacts.example.com
    --summary-path artifacts
    # S3 key:  s3://bucket/v3/artifacts/{run_id}-linux/...
    # Summary: https://artifacts.example.com/artifacts/{run_id}-linux/...

This script runs after building TheRock, where this script does:
  1. Create log archives
  2. (optional) upload artifacts
  3. (optional) upload logs
  4. (optional) add links to GitHub job summary

In the case that a CI build fails, this step will always upload available logs and artifacts.

For AWS credentials to upload, reach out to the #rocm-ci channel in the AMD Developer Community Discord
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import platform
import sys
import tarfile

from github_actions_api import gha_append_step_summary, str2bool

THEROCK_DIR = Path(__file__).resolve().parent.parent.parent
PLATFORM = platform.system().lower()

# Add build_tools to path for _therock_utils imports.
sys.path.insert(0, str(THEROCK_DIR / "build_tools"))
from _therock_utils.workflow_outputs import WorkflowOutputRoot
from _therock_utils.storage_backend import StorageBackend, create_storage_backend


def log(*args):
    print(*args)
    sys.stdout.flush()


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


def _get_pyzstd():
    """Lazy import pyzstd with helpful error message."""
    try:
        import pyzstd

        return pyzstd
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "pyzstd is required for zstd compression. "
            "Install it with: pip install pyzstd"
        )


def create_ccache_log_archive(build_dir: Path):
    """Archive the ccache log subdirectory into a zstd-compressed tarball.

    ccache.log can be hundreds of MB (verbose per-invocation trace) but
    compresses ~13x with zstd. The raw logs in logs/ccache/ are excluded from
    upload (see upload_logs); this archive provides the compressed version.
    """
    ccache_dir = build_dir / "logs" / "ccache"
    if not ccache_dir.is_dir():
        return

    found_files = sorted(f for f in ccache_dir.iterdir() if f.is_file())
    if not found_files:
        return

    pyzstd = _get_pyzstd()
    archive_path = build_dir / "logs" / "ccache_logs.tar.zst"
    with pyzstd.ZstdFile(archive_path, mode="wb") as zst:
        with tarfile.open(mode="w|", fileobj=zst) as tar:
            for file_path in found_files:
                tar.add(file_path, arcname=file_path.name)
                log(f"[+] Archived ccache log: {file_path.name}")

    archive_size = archive_path.stat().st_size
    log(
        f"[INFO] Created {archive_path.name} "
        f"({archive_size // 1024 // 1024}MB from {len(found_files)} files)"
    )


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
    #   - therock-build-prof/ subdirectory (resource profiling)
    # index.html is generated server-side by build_tools/generate_s3_index.py after upload.
    # Content-type is inferred per file by the backend.
    # Exclude raw ccache logs — they're uploaded compressed as ccache_logs.tar.zst.
    log("Uploading logs")
    backend.upload_directory(
        log_dir, output_root.log_dir(artifact_group), exclude=["ccache/**/*"]
    )

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
    skip_manifest: bool = False,
    summary_base_url: str = "",
    summary_path: str = "",
):
    log(f"Adding links to job summary for {output_root.prefix}")

    def _url(loc) -> str:
        if summary_base_url:
            return loc.cdn_url(
                base_url=summary_base_url,
                s3_prefix=output_root.path_prefix,
                cdn_prefix=summary_path,
            )
        return loc.https_url

    log_index_url = _url(output_root.log_index(artifact_group))
    gha_append_step_summary(f"[Build Logs]({log_index_url})")

    observability_path = build_dir / "logs" / "build_observability.html"
    if observability_path.is_file():
        analysis_url = _url(output_root.build_observability(artifact_group))
        gha_append_step_summary(f"[Build Observability]({analysis_url})")
    else:
        log("[INFO] Build Observability: Not generated")

    # Only add artifact links if the job not failed
    if not job_status or job_status == "success":
        artifact_url = _url(output_root.artifact_index(artifact_group))
        gha_append_step_summary(f"[Artifacts]({artifact_url})")

    if not skip_manifest:
        manifest_url = _url(output_root.manifest(artifact_group))
        gha_append_step_summary(f"[TheRock Manifest]({manifest_url})")


def run(args):
    log("Creating Ninja log archive")
    log("--------------------------")
    create_ninja_log_archive(args.build_dir)

    log("Creating ccache log archive")
    log("---------------------------")
    create_ccache_log_archive(args.build_dir)

    if not args.upload:
        return

    output_root = WorkflowOutputRoot.from_workflow_run(
        run_id=args.run_id,
        platform=PLATFORM,
        bucket_override=args.artifacts_bucket or None,
        path_prefix=args.path_prefix,
    )
    backend = create_storage_backend(staging_dir=args.output_dir, dry_run=args.dry_run)

    # Upload artifacts only if the job not failed
    if not args.job_status or args.job_status == "success":
        log("Upload build artifacts")
        log("----------------------")
        upload_artifacts(args.artifact_group, args.build_dir, output_root, backend)

    log("Upload log")
    log("----------")
    upload_logs(args.artifact_group, args.build_dir, output_root, backend)

    if not args.skip_manifest:
        log("Upload manifest")
        log("----------------")
        upload_manifest(args.artifact_group, args.build_dir, output_root, backend)

    log("Write github actions build summary")
    log("--------------------")
    write_gha_build_summary(
        args.artifact_group,
        args.build_dir,
        output_root,
        args.job_status,
        skip_manifest=args.skip_manifest,
        summary_base_url=args.summary_base_url,
        summary_path=args.summary_path,
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
    parser.add_argument(
        "--artifacts-bucket",
        type=str,
        default="",
        help="Override the S3 bucket for artifacts. If unset, the bucket is determined from repository/fork/release_type. Bucket permissions are enforced by AWS IAM.",
    )
    parser.add_argument(
        "--path-prefix",
        type=str,
        default="",
        help="S3 key prefix for all outputs (e.g. 'v3/artifacts').",
    )
    parser.add_argument(
        "--summary-base-url",
        type=str,
        default="",
        help="CDN base URL for job summary links (e.g. 'https://artifacts.example.com'). When set, summary links use this instead of the raw S3 URL.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default="",
        help="CDN path prefix for summary links, corresponding to --path-prefix on S3 (e.g. 'artifacts').",
    )
    parser.add_argument(
        "--skip-manifest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip manifest upload and omit the manifest link from the job summary.",
    )
    args = parser.parse_args()

    # Check preconditions for provided arguments before proceeding.

    args.path_prefix = args.path_prefix.rstrip("/")
    args.summary_base_url = args.summary_base_url.rstrip("/")
    args.summary_path = args.summary_path.rstrip("/")

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
