#!/usr/bin/env python

"""Fetches artifacts from S3.

The install_rocm_from_artifacts.py script builds on top of this script to both
download artifacts then unpack them into a usable install directory.

Example usage (using https://github.com/ROCm/TheRock/actions/runs/15685736080):
  pip install boto3
  python build_tools/fetch_artifacts.py \
    --run-id 15685736080 --target gfx110X-dgpu --output-dir ~/.therock/artifacts_15685736080

Or, to fetch _all_ artifacts and not just a subset (this is safest for packaging
workflows where dependencies may not be accurately modeled, at the cost of
additional disk space):
  python build_tools/fetch_artifacts.py \
    --run-id 15685736080 --target gfx110X-dgpu --output-dir ~/.therock/artifacts_15685736080 \
    --all

Alternatively, include/exclude regular expressions can be given to control what
is downloaded (this implies --all):
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
from dataclasses import dataclass
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tarfile
import time
import warnings
from urllib3.exceptions import InsecureRequestWarning

from _therock_utils.artifacts import ArtifactPopulator


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

THEROCK_DIR = Path(__file__).resolve().parent.parent

# Importing build_artifact_upload.py
sys.path.append(str(THEROCK_DIR / "build_tools" / "github_actions"))
from _therock_utils.artifacts import ArtifactName
from github_actions_utils import *

GENERIC_VARIANT = "generic"
PLATFORM = platform.system().lower()


# TODO(geomin12): switch out logging library
def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def retrieve_s3_artifacts(run_id, amdgpu_family):
    """Checks that the AWS S3 bucket exists and returns artifact names."""
    EXTERNAL_REPO, BUCKET = retrieve_bucket_info()
    s3_directory_path = f"{EXTERNAL_REPO}{run_id}-{PLATFORM}/"
    page_iterator = paginator.paginate(Bucket=BUCKET, Prefix=s3_directory_path)
    data = set()
    for page in page_iterator:
        if "Contents" in page:
            for artifact in page["Contents"]:
                artifact_key = artifact["Key"]
                if (
                    "sha256sum" not in artifact_key
                    and "tar.xz" in artifact_key
                    and (amdgpu_family in artifact_key or "generic" in artifact_key)
                ):
                    file_name = artifact_key.split("/")[-1]
                    data.add(file_name)
    return data


@dataclass
class ArtifactDownloadRequest:
    """Information about a request to download an artifact to a local path."""

    artifact_key: str
    bucket: str
    output_path: Path

    def __str__(self):
        return f"{self.bucket}:{self.artifact_key}"


def get_bucket_url(run_id: str):
    external_repo, bucket = retrieve_bucket_info()
    return f"https://{bucket}.s3.us-east-2.amazonaws.com/{external_repo}{run_id}-{PLATFORM}"


def collect_artifacts_download_requests(
    artifact_names: list[str],
    run_id: str,
    output_dir: Path,
    variant: str,
    existing_artifacts: set[str],
) -> list[ArtifactDownloadRequest]:
    """Collects S3 artifact URLs to execute later in parallel."""
    artifacts_to_retrieve = []
    EXTERNAL_REPO, BUCKET = retrieve_bucket_info()
    s3_key_path = f"{EXTERNAL_REPO}{run_id}-{PLATFORM}"
    for artifact_name in artifact_names:
        file_name = f"{artifact_name}_{variant}.tar.xz"
        # If artifact does exist in s3 bucket
        if file_name in existing_artifacts:
            artifacts_to_retrieve.append(
                ArtifactDownloadRequest(
                    artifact_key=f"{s3_key_path}/{file_name}",
                    bucket=BUCKET,
                    output_path=output_dir / file_name,
                )
            )

    return artifacts_to_retrieve


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


def filter_all_artifacts(
    run_id: str,
    target: str,
    output_dir: Path,
    s3_artifacts: set[str],
) -> list[ArtifactDownloadRequest]:
    """Filters to all available artifacts."""
    artifacts_to_retrieve = []
    EXTERNAL_REPO, BUCKET = retrieve_bucket_info()
    s3_key_path = f"{EXTERNAL_REPO}{run_id}-{PLATFORM}"

    for artifact in sorted(list(s3_artifacts)):
        an = ArtifactName.from_filename(artifact)
        if an.target_family != "generic" and target != an.target_family:
            continue

        artifacts_to_retrieve.append(
            ArtifactDownloadRequest(
                artifact_key=f"{s3_key_path}/{artifact}",
                bucket=BUCKET,
                output_path=output_dir / artifact,
            )
        )
    return artifacts_to_retrieve


def get_postprocess_mode(args) -> str | None:
    """Returns 'extract', 'flatten' or None (default is 'extract')."""
    if args.flatten:
        return "flatten"
    if args.no_extract:
        return None
    return "extract"


def filter_base_artifacts(
    args: argparse.Namespace,
    run_id: str,
    output_dir: Path,
    s3_artifacts: set[str],
) -> list[ArtifactDownloadRequest]:
    """Filters TheRock base artifacts."""
    base_artifacts = [
        "core-runtime_run",
        "core-runtime_lib",
        "sysdeps_lib",
        "base_run",
        "base_lib",
        "amd-llvm_run",
        "amd-llvm_lib",
        "core-hip_lib",
        "core-hip_dev",
        "rocprofiler-sdk_lib",
        "host-suite-sparse_lib",
    ]
    if args.blas:
        base_artifacts.append("host-blas_lib")

    artifacts_to_retrieve = collect_artifacts_download_requests(
        base_artifacts, run_id, output_dir, GENERIC_VARIANT, s3_artifacts
    )
    return artifacts_to_retrieve


def filter_enabled_artifacts(
    args: argparse.Namespace,
    target: str,
    run_id: str,
    output_dir: Path,
    s3_artifacts: set[str],
) -> list[ArtifactDownloadRequest]:
    """Filters TheRock artifacts using based on the enabled arguments.

    If no artifacts have been collected, we assume that we want to install the default subset.
    If `args.tests` have been enabled, we also collect test artifacts.
    """
    artifact_paths = []
    all_artifacts = ["blas", "fft", "miopen", "prim", "rand"]
    # RCCL is disabled for Windows
    if PLATFORM != "windows":
        all_artifacts.append("rccl")

    if args.blas:
        artifact_paths.append("blas")
    if args.fft:
        artifact_paths.append("fft")
    if args.miopen:
        artifact_paths.append("miopen")
    if args.prim:
        artifact_paths.append("prim")
    if args.rand:
        artifact_paths.append("rand")
    if args.rccl and PLATFORM != "windows":
        artifact_paths.append("rccl")

    enabled_artifacts = []

    # In the case that no library arguments were passed and base_only args is false, we install all artifacts
    if not artifact_paths and not args.base_only:
        artifact_paths = all_artifacts

    for base_path in artifact_paths:
        enabled_artifacts.append(f"{base_path}_lib")
        if args.tests:
            enabled_artifacts.append(f"{base_path}_test")

    artifacts_to_retrieve = collect_artifacts_download_requests(
        enabled_artifacts, run_id, output_dir, target, s3_artifacts
    )
    return artifacts_to_retrieve


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
    run_id = args.run_id
    target = args.target
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    s3_artifacts = retrieve_s3_artifacts(run_id, target)
    if not s3_artifacts:
        log(f"S3 artifacts for {run_id} does not exist. Exiting...")
        sys.exit(1)

    # Filter the artifacts we will retrieve.
    artifacts_to_retrieve: list[ArtifactDownloadRequest] | None = None
    if args.include:
        args.all = True
    if args.all:
        artifacts_to_retrieve = filter_all_artifacts(
            run_id, target, output_dir, s3_artifacts
        )
    else:
        artifacts_to_retrieve = filter_base_artifacts(
            args, run_id, output_dir, s3_artifacts
        )
        if not args.base_only:
            artifacts_to_retrieve.extend(
                filter_enabled_artifacts(args, target, run_id, output_dir, s3_artifacts)
            )

    # Include/exclude filtering.
    def _should_include(artifact: ArtifactDownloadRequest) -> bool:
        if not args.include:
            return True
        # If includes, then one include must match.
        for include in args.include:
            pattern = re.compile(include)
            if pattern.search(artifact.artifact_key):
                break
        else:
            return False
        # If excludes, then no excludes must match.
        if args.exclude:
            for exclude in args.exclude:
                pattern = re.compile(exclude)
                if pattern.search(artifact.artifact_key):
                    return False
        return True

    artifacts_to_retrieve = [a for a in artifacts_to_retrieve if _should_include(a)]

    joined_artifact_summary = "\n  ".join([str(item) for item in artifacts_to_retrieve])
    log(f"Downloading in parallel:\n  {joined_artifact_summary}\n")

    # Download and extract in parallel.
    postprocess_mode = get_postprocess_mode(args)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.download_concurrency
    ) as download_executor, concurrent.futures.ThreadPoolExecutor(
        max_workers=args.extract_concurrency
    ) as extract_executor:
        download_futures = [
            download_executor.submit(download_artifact, req)
            for req in artifacts_to_retrieve
        ]
        extract_futures: list[concurrent.futures.Future] = []
        for download_future in concurrent.futures.as_completed(download_futures):
            download_result = download_future.result(timeout=60)
            if postprocess_mode:
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
        "--run-id",
        type=str,
        required=True,
        help="GitHub run ID to retrieve artifacts from",
    )

    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target variant for specific GPU target",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default="build/artifacts",
        help="Output path for fetched artifacts, defaults to `build/artifacts/` as in source builds",
    )

    # Post processing mode
    postprocess_p = parser.add_mutually_exclusive_group()
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

    parser.add_argument(
        "--delete-after-extract",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Delete archive files after extraction",
    )
    artifacts_group = parser.add_argument_group("artifacts_group")
    artifacts_group.add_argument(
        "--all",
        default=False,
        help="Include all artifacts",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "include",
        nargs="*",
        help="Regular expression patterns of artifacts to include (implies --all): "
        "if supplied one pattern must match for an artifact to be included",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        help="Regular expression patterns of artifacts to exclude",
    )

    artifacts_group.add_argument(
        "--blas",
        default=False,
        help="Include 'blas' artifacts",
        action=argparse.BooleanOptionalAction,
    )
    artifacts_group.add_argument(
        "--fft",
        default=False,
        help="Include 'fft' artifacts",
        action=argparse.BooleanOptionalAction,
    )
    artifacts_group.add_argument(
        "--miopen",
        default=False,
        help="Include 'miopen' artifacts",
        action=argparse.BooleanOptionalAction,
    )
    artifacts_group.add_argument(
        "--prim",
        default=False,
        help="Include 'prim' artifacts",
        action=argparse.BooleanOptionalAction,
    )
    artifacts_group.add_argument(
        "--rand",
        default=False,
        help="Include 'rand' artifacts",
        action=argparse.BooleanOptionalAction,
    )
    artifacts_group.add_argument(
        "--rccl",
        default=False,
        help="Include 'rccl' artifacts",
        action=argparse.BooleanOptionalAction,
    )
    artifacts_group.add_argument(
        "--tests",
        default=False,
        help="Include all test artifacts for enabled libraries",
        action=argparse.BooleanOptionalAction,
    )
    artifacts_group.add_argument(
        "--base-only", help="Include only base artifacts", action="store_true"
    )

    args = parser.parse_args(argv)

    if (args.all or args.include) and (
        args.blas
        or args.fft
        or args.miopen
        or args.prim
        or args.rand
        or args.rccl
        or args.tests
        or args.base_only
    ):
        parser.error("--all cannot be set together with artifact group options")

    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
