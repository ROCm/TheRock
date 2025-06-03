#!/usr/bin/env python
# This script provides a somewhat dynamic way to
# retrieve artifacts from s3

# NOTE: This script currently only retrieves the requested artifacts,
# but those artifacts may not have all required dependencies.

import argparse
import concurrent.futures
from html.parser import HTMLParser
from pathlib import Path
import platform
from shutil import copyfileobj
import sys
import urllib.request

THEROCK_DIR = Path(__file__).resolve().parent.parent

# Importing build_artifact_upload.py
sys.path.append(str(THEROCK_DIR / "build_tools" / "github_actions"))
from upload_build_artifacts import retrieve_bucket_info

GENERIC_VARIANT = "generic"
PLATFORM = platform.system().lower()


class FetchArtifactException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class ArtifactNotFoundExeption(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class IndexPageParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.files = []
        self.is_file_data = False

    def handle_starttag(self, tag, attrs):
        if tag == "span":
            for attr_name, attr_value in attrs:
                if attr_name == "class" and attr_value == "name":
                    self.is_file_data = True
                    break

    def handle_data(self, data):
        if self.is_file_data:
            self.files.append(data)
            self.is_file_data = False


# TODO(geomin12): switch out logging library
def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def retrieve_s3_artifacts(run_id, amdgpu_family):
    """Checks that the AWS S3 bucket exists and returns artifact names."""
    EXTERNAL_REPO, BUCKET = retrieve_bucket_info()
    BUCKET_URL = f"https://{BUCKET}.s3.amazonaws.com/{EXTERNAL_REPO}{run_id}-{PLATFORM}"
    index_page_url = f"{BUCKET_URL}/index-{amdgpu_family}.html"
    log(f"Retrieving artifacts from {index_page_url}")
    request = urllib.request.Request(index_page_url)
    try:
        with urllib.request.urlopen(request) as response:
            # from the S3 index page, we search for artifacts inside the a tags "<span class='name'>{TAR_NAME}</span>"
            parser = IndexPageParser()
            parser.feed(str(response.read()))
            data = set()
            for artifact in parser.files:
                # We only want to get .tar.xz files, not .tar.xz.sha256sum
                if "sha256sum" not in artifact and "tar.xz" in artifact:
                    data.add(artifact)
            return data
    except urllib.request.HTTPError as err:
        if err.code == 404:
            raise ArtifactNotFoundExeption(
                f"No artifacts found for {run_id}-{PLATFORM}. Exiting..."
            )
        else:
            raise FetchArtifactException(
                f"Error when retrieving S3 bucket {run_id}-{PLATFORM}/index-{amdgpu_family}.html. Exiting..."
            )


def collect_artifacts_urls(
    artifacts: list[str],
    run_id: str,
    build_dir: str,
    variant: str,
    existing_artifacts: set[str],
) -> list[str]:
    """Collects S3 artifact URLs to execute later in parallel."""
    EXTERNAL_REPO, BUCKET = retrieve_bucket_info()
    BUCKET_URL = f"https://{BUCKET}.s3.us-east-2.amazonaws.com/{EXTERNAL_REPO}{run_id}-{PLATFORM}"
    artifacts_to_retrieve = []
    for artifact in artifacts:
        file_name = f"{artifact}_{variant}.tar.xz"
        # If artifact does exist in s3 bucket
        if file_name in existing_artifacts:
            # Tuple of (FILE_PATH_TO_WRITE, S3_ARTIFACT_URL)
            artifacts_to_retrieve.append(
                (
                    f"{build_dir}/{file_name}",
                    f"{BUCKET_URL}/{file_name}",
                )
            )

    return artifacts_to_retrieve


def urllib_retrieve_artifact(artifact):
    output_path, artifact_url = artifact
    log(f"++ Downloading from {artifact_url} to {output_path}")
    with urllib.request.urlopen(artifact_url) as in_stream, open(
        output_path, "wb"
    ) as out_file:
        copyfileobj(in_stream, out_file)
    log(f"++ Download complete for {output_path}")


def parallel_exec_commands(artifacts):
    """Runs parallelized urllib calls using a thread pool executor"""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(urllib_retrieve_artifact, artifact)
            for artifact in artifacts
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result(timeout=60)


def retrieve_base_artifacts(args, run_id, build_dir, s3_artifacts):
    """Retrieves TheRock base artifacts using urllib."""
    base_artifacts = [
        "core-runtime_run",
        "core-runtime_lib",
        "sysdeps_lib",
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

    artifacts_to_retrieve = collect_artifacts_urls(
        base_artifacts, run_id, build_dir, GENERIC_VARIANT, s3_artifacts
    )
    parallel_exec_commands(artifacts_to_retrieve)


def retrieve_enabled_artifacts(args, target, run_id, build_dir, s3_artifacts):
    """Retrieves TheRock artifacts using urllib, based on the enabled arguments.

    If no artifacts have been collected, we assume that we want to install all artifacts
    If `args.tests` have been enabled, we also collect test artifacts
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

    artifacts_to_retrieve = collect_artifacts_urls(
        enabled_artifacts, run_id, build_dir, target, s3_artifacts
    )
    parallel_exec_commands(artifacts_to_retrieve)


def run(args):
    run_id = args.run_id
    target = args.target
    build_dir = args.build_dir
    s3_artifacts = retrieve_s3_artifacts(run_id, target)
    if not s3_artifacts:
        print(f"S3 artifacts for {run_id} does not exist. Exiting...")
        return
    retrieve_base_artifacts(args, run_id, build_dir, s3_artifacts)
    if not args.base_only:
        retrieve_enabled_artifacts(args, target, run_id, build_dir, s3_artifacts)


def main(argv):
    parser = argparse.ArgumentParser(prog="fetch_artifacts")
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
        "--build-dir",
        type=str,
        default="build/artifacts",
        help="Path to the artifact build directory",
    )

    artifacts_group = parser.add_argument_group("artifacts_group")
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
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
