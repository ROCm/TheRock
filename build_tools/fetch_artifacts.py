#!/usr/bin/env python

"""Fetches artifacts from S3.

The install_rocm_from_artifacts.py script builds on top of this script to both
download artifacts then unpack them into a usable install directory.

Example usage (using https://github.com/ROCm/TheRock/actions/runs/15685736080):
  pip install boto3
  python build_tools/fetch_artifacts.py \
    --run-id 15685736080 --target gfx110X-dgpu --output-dir ~/.therock/artifacts_15685736080

Include/exclude regular expressions can be given to control what is downloaded:
  python build_tools/fetch_artifacts.py \
    --run-id 15685736080 --target gfx110X-dgpu --output-dir ~/.therock/artifacts_15685736080 \
    amd-llvm base 'core-(hip|runtime)' sysdeps \
    --exclude _dbg_

This will process artifacts that match any of the include patterns and do not
match any of the exclude patterns.

Note this module will respect:
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_SESSION_TOKEN
if and only if all are specified in the environment to connect with S3.
If unspecified, we will create an anonymous boto file that can only acccess public artifacts.
"""

import argparse
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import concurrent.futures
from dataclasses import dataclass, field
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tarfile
import time
from urllib3.exceptions import InsecureRequestWarning
import warnings

from _therock_utils.artifacts import ArtifactName, ArtifactPopulator
from github_actions.github_actions_utils import retrieve_bucket_info


warnings.filterwarnings("ignore", category=InsecureRequestWarning)

_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
_session_token = os.environ.get("AWS_SESSION_TOKEN")

# Create S3 client leveraging AWS credentials if available.
if None not in (_access_key_id, _secret_access_key, _session_token):
    s3_client = boto3.client(
        "s3",
        verify=False,
        aws_access_key_id=_access_key_id,
        aws_secret_access_key=_secret_access_key,
        aws_session_token=_session_token,
    )
else:
    # Otherwise use anonymous boto file.
    s3_client = boto3.client(
        "s3",
        verify=False,
        config=Config(max_pool_connections=100, signature_version=UNSIGNED),
    )

paginator = s3_client.get_paginator("list_objects_v2")


# TODO(geomin12): switch out logging library
def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


# TODO: move into github_actions_utils.py?
@dataclass
class BucketMetadata:
    """Metadata for a workflow run's artifacts in an AWS S3 bucket."""

    external_repo: str
    bucket: str
    workflow_run_id: str
    platform: str
    s3_key_path: str = field(init=False)

    def __post_init__(self):
        self.s3_key_path = f"{self.external_repo}{self.workflow_run_id}-{self.platform}"


def list_s3_artifacts(bucket_info: BucketMetadata, amdgpu_family: str) -> set[str]:
    """Checks that the AWS S3 bucket exists and returns artifact names."""
    s3_key_path = bucket_info.s3_key_path
    log(
        f"Retrieving S3 artifacts for {bucket_info.workflow_run_id} in '{bucket_info.bucket}' at '{s3_key_path}'"
    )

    page_iterator = paginator.paginate(Bucket=bucket_info.bucket, Prefix=s3_key_path)
    data = set()
    for page in page_iterator:
        if not "Contents" in page:
            continue

        for artifact in page["Contents"]:
            artifact_key = artifact["Key"]
            if (
                "sha256sum" not in artifact_key
                and "tar.xz" in artifact_key
                and (amdgpu_family in artifact_key or "generic" in artifact_key)
            ):
                file_name = artifact_key.split("/")[-1]
                data.add(file_name)
    if not data:
        log(f"Found no S3 artifacts for {bucket_info.run_id} at '{s3_key_path}'")
    return data


def filter_artifacts(
    artifacts: set[str], includes: list[str], excludes: list[str]
) -> set[str]:
    """Filters artifacts based on include and exclude regex lists"""

    def _should_include(artifact_name: str) -> bool:
        if not includes and not excludes:
            return True

        # If includes, then one include must match.
        if includes:
            for include in includes:
                pattern = re.compile(include)
                if pattern.search(artifact_name):
                    break
            else:
                return False

        # If excludes, then no excludes must match.
        if excludes:
            for exclude in excludes:
                pattern = re.compile(exclude)
                if pattern.search(artifact_name):
                    return False

        # Included and not excluded.
        return True

    return {a for a in artifacts if _should_include(a)}


@dataclass
class ArtifactDownloadRequest:
    """Information about a request to download an artifact to a local path."""

    artifact_key: str
    bucket: str
    output_path: Path

    def __str__(self):
        return f"{self.bucket}:{self.artifact_key}"


def download_artifact(
    artifact_download_request: ArtifactDownloadRequest,
) -> ArtifactDownloadRequest:
    MAX_RETRIES = 3
    BASE_DELAY = 3  # seconds
    for attempt in range(MAX_RETRIES):
        try:
            artifact_key = artifact_download_request.artifact_key
            bucket = artifact_download_request.bucket
            output_path = artifact_download_request.output_path
            log(f"++ Downloading {artifact_key} to {output_path}")
            with open(output_path, "wb") as f:
                s3_client.download_fileobj(bucket, artifact_key, f)
            log(f"++ Download complete for {output_path}")
            return artifact_download_request
        except Exception as e:
            log(f"++ Error downloading {artifact_key}: {e}")
            if attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY * (2**attempt)
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                log(
                    f"++ Failed downloading from {artifact_key} after {MAX_RETRIES} retries"
                )


def download_artifacts(artifact_download_requests: list[ArtifactDownloadRequest]):
    """Downloads artifacts in parallel using a thread pool executor."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(download_artifact, artifact_download_request)
            for artifact_download_request in artifact_download_requests
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result(timeout=60)


def get_artifact_download_requests(
    bucket_info: BucketMetadata,
    s3_artifacts: set[str],
    output_dir: Path,
) -> list[ArtifactDownloadRequest]:
    """Gets artifact download requests from requested artifacts."""
    artifacts_to_download = []

    for artifact in sorted(list(s3_artifacts)):
        artifacts_to_download.append(
            ArtifactDownloadRequest(
                artifact_key=f"{bucket_info.s3_key_path}/{artifact}",
                bucket=bucket_info.bucket,
                output_path=output_dir / artifact,
            )
        )
    return artifacts_to_download


def get_postprocess_mode(args) -> str | None:
    """Returns 'extract', 'flatten' or None (default is 'extract')."""
    if args.flatten:
        return "flatten"
    if args.no_extract:
        return None
    return "extract"


def extract_artifact(
    artifact: ArtifactDownloadRequest, *, delete_archive: bool, postprocess_mode: str
):
    # Get (for example) 'amd-llvm_lib_generic' from '/path/to/amd-llvm_lib_generic.tar.xz'
    # We can't just use .stem since that only removes the last extension.
    #   1. .name gets us 'amd-llvm_lib_generic.tar.xz'
    #   2. .partition('.') gets (before, sep, after), discard all but 'before'
    archive_file = artifact.output_path
    artifact_name, *_ = archive_file.name.partition(".")

    if postprocess_mode == "extract":
        output_dir = archive_file.parent / artifact_name
        if output_dir.exists():
            shutil.rmtree(output_dir)
        with tarfile.TarFile.open(archive_file, mode="r:xz") as tf:
            log(f"++ Extracting '{archive_file.name}' to '{artifact_name}'")
            tf.extractall(archive_file.parent / artifact_name, filter="tar")
    elif postprocess_mode == "flatten":
        output_dir = archive_file.parent
        log(f"++ Flattening '{archive_file.name}' to '{artifact_name}'")
        flattener = ArtifactPopulator(
            output_path=output_dir, verbose=True, flatten=True
        )
        flattener(archive_file)
    else:
        raise AssertionError(f"Unhandled postprocess_mode = {postprocess_mode}")

    if delete_archive:
        archive_file.unlink()


def run(args):
    run_github_repo = args.run_github_repo
    run_id = args.run_id
    target = args.target
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    external_repo, bucket = retrieve_bucket_info(
        github_repository=run_github_repo,
        workflow_run_id=run_id,
    )
    bucket_info = BucketMetadata(
        external_repo=external_repo,
        bucket=bucket,
        workflow_run_id=run_id,
        platform=args.platform,
    )

    # Lookup which artifacts exist in the bucket.
    # Note: this currently does not check that all requested artifacts
    # (via include patterns) do exist, so this may silently fail to fetch
    # expected files.
    s3_artifacts = list_s3_artifacts(bucket_info=bucket_info, amdgpu_family=target)
    if not s3_artifacts:
        log(f"No matching artifacts for {run_id} exist. Exiting...")
        sys.exit(1)

    # Include/exclude filtering.
    s3_artifacts_filtered = filter_artifacts(s3_artifacts, args.include, args.exclude)
    if not s3_artifacts_filtered:
        log(f"Filtering artifacts for {run_id} resulted in an empty set. Exiting...")
        sys.exit(1)

    artifacts_to_download = get_artifact_download_requests(
        bucket_info=bucket_info,
        s3_artifacts=s3_artifacts_filtered,
        output_dir=output_dir,
    )

    download_summary = "\n  ".join([str(item) for item in artifacts_to_download])
    log(f"\nFiltered artifacts to download:\n  {download_summary}\n")

    if args.dry_run:
        log("Skipping downloads since --dry-run was set")
        return

    # Download and extract in parallel.
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.download_concurrency
    ) as download_executor:
        download_futures = [
            download_executor.submit(download_artifact, req)
            for req in artifacts_to_download
        ]

        postprocess_mode = get_postprocess_mode(args)
        if not postprocess_mode:
            # No postprocessing to do, wait on downloads then return.
            [f.result() for f in download_futures]
            return

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.extract_concurrency
        ) as extract_executor:
            extract_futures: list[concurrent.futures.Future] = []
            for download_future in concurrent.futures.as_completed(download_futures):
                download_result = download_future.result(timeout=60)
                extract_futures.append(
                    extract_executor.submit(
                        extract_artifact,
                        download_result,
                        delete_archive=args.delete_after_extract,
                        postprocess_mode=postprocess_mode,
                    )
                )

            [f.result() for f in extract_futures]


def main(argv):
    parser = argparse.ArgumentParser(prog="fetch_artifacts")

    filter_group = parser.add_argument_group("Artifact filtering")
    filter_group.add_argument(
        "include",
        nargs="*",
        help="Regular expression patterns of artifacts to include: "
        "if supplied one pattern must match for an artifact to be included",
    )
    filter_group.add_argument(
        "--exclude",
        nargs="*",
        help="Regular expression patterns of artifacts to exclude",
    )
    filter_group.add_argument(
        "--platform",
        type=str,
        choices=["linux", "windows"],
        default=platform.system().lower(),
        help="Platform to download artifacts for (matches artifact folder name suffixes in S3)",
    )
    filter_group.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target variant for specific GPU target",
    )

    parser.add_argument(
        "--download-concurrency",
        type=int,
        default=10,
        help="Number of concurrent download jobs to execute at once",
    )
    parser.add_argument(
        "--extract-concurrency",
        type=int,
        help="Number of extract jobs to execute at once (defaults to python VM defaults for CPU tasks)",
    )
    parser.add_argument(
        "--run-github-repo",
        type=str,
        help="GitHub repository for --run-id. If omitted, this is inferred from the GITHUB_REPOSITORY env var or defaults to ROCm/TheRock",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="GitHub run ID to retrieve artifacts from",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="build/artifacts",
        help="Output path for fetched artifacts, defaults to `build/artifacts/` as in source builds",
    )
    parser.add_argument(
        "--dry-run",
        default=False,
        help="If set, will only log which artifacts would be fetched without downloading or extracting",
        action=argparse.BooleanOptionalAction,
    )

    postprocess_group = parser.add_argument_group("Postprocessing")
    postprocess_p = postprocess_group.add_mutually_exclusive_group()
    postprocess_p.add_argument(
        "--no-extract",
        default=False,
        action="store_true",
        help="Do no extraction or flattening",
    )
    postprocess_p.add_argument(
        "--extract",
        default=False,
        action="store_true",
        help="Extract files after fetching them",
    )
    postprocess_p.add_argument(
        "--flatten",
        default=False,
        action="store_true",
        help="Flattens artifacts after fetching them",
    )
    postprocess_group.add_argument(
        "--delete-after-extract",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Delete archive files after extraction",
    )

    args = parser.parse_args(argv)

    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
