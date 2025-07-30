#!/usr/bin/env python

"""Fetches artifacts from S3.

The install_rocm_from_artifacts.py script builds on top of this script to both
download artifacts then unpack them into a usable install directory.

Example usage (using https://github.com/ROCm/TheRock/actions/runs/15685736080):
  mkdir -p ~/.therock/artifacts_15685736080
  python build_tools/fetch_artifacts.py \
    --run-id 15685736080 --target gfx110X-dgpu --output-dir ~/.therock/artifacts_15685736080

Or, to fetch _all_ artifacts and not just a subset (this is safest for packaging
workflows where dependencies may not be accurately modeled, at the cost of
additional disk space):
  mkdir -p ~/.therock/artifacts_15685736080
  python build_tools/fetch_artifacts.py \
    --run-id 15685736080 --target gfx110X-dgpu --output-dir ~/.therock/artifacts_15685736080 \
    --all
"""

import argparse
import concurrent.futures
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
import platform
from shutil import copyfileobj
import sys
import tarfile
import time
import urllib.request

THEROCK_DIR = Path(__file__).resolve().parent.parent

# Importing build_artifact_upload.py
sys.path.append(str(THEROCK_DIR / "build_tools" / "github_actions"))
from upload_build_artifacts import retrieve_bucket_info
from _therock_utils.artifacts import ArtifactName

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
    # TODO(scotttodd): hint when amdgpu_family is wrong/missing (root index for all families/platforms)?
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


@dataclass
class ArtifactDownloadRequest:
    """Information about a request to download an artifact to a local path."""

    artifact_url: str
    output_path: Path


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
    bucket_url = get_bucket_url(run_id)

    artifacts_to_retrieve = []
    for artifact_name in artifact_names:
        file_name = f"{artifact_name}_{variant}.tar.xz"
        # If artifact does exist in s3 bucket
        if file_name in existing_artifacts:
            artifacts_to_retrieve.append(
                ArtifactDownloadRequest(
                    artifact_url=f"{bucket_url}/{file_name}",
                    output_path=output_dir / file_name,
                )
            )

    return artifacts_to_retrieve


def download_artifact(artifact_download_request: ArtifactDownloadRequest):
    MAX_RETRIES = 3
    BASE_DELAY = 3  # seconds
    for attempt in range(MAX_RETRIES):
        try:
            artifact_url = artifact_download_request.artifact_url
            output_path = artifact_download_request.output_path
            log(f"++ Downloading from {artifact_url} to {output_path}")
            with urllib.request.urlopen(artifact_url) as in_stream, open(
                output_path, "wb"
            ) as out_file:
                copyfileobj(in_stream, out_file)
            log(f"++ Download complete for {output_path}")
        except Exception as e:
            log(f"++ Error downloading from {artifact_url}: {e}")
            if attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY * (2**attempt)
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                log(
                    f"++ Failed downloading from {artifact_url} after {MAX_RETRIES} retries"
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


def retrieve_all_artifacts(
    run_id: str,
    target: str,
    output_dir: Path,
    s3_artifacts: set[str],
):
    """Retrieves all available artifacts."""

    bucket_url = get_bucket_url(run_id)
    artifacts_to_retrieve = []
    for artifact in sorted(list(s3_artifacts)):
        an = ArtifactName.from_filename(artifact)
        if an.target_family != "generic" and target != an.target_family:
            continue

        artifacts_to_retrieve.append(
            ArtifactDownloadRequest(
                artifact_url=f"{bucket_url}/{artifact}",
                output_path=output_dir / artifact,
            )
        )

    download_artifacts(artifacts_to_retrieve)


def retrieve_base_artifacts(
    args: argparse.Namespace,
    run_id: str,
    output_dir: Path,
    s3_artifacts: set[str],
):
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

    artifacts_to_retrieve = collect_artifacts_download_requests(
        base_artifacts, run_id, output_dir, GENERIC_VARIANT, s3_artifacts
    )
    download_artifacts(artifacts_to_retrieve)


def retrieve_enabled_artifacts(
    args: argparse.Namespace,
    target: str,
    run_id: str,
    output_dir: Path,
    s3_artifacts: set[str],
):
    """Retrieves TheRock artifacts using urllib, based on the enabled arguments.

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
    download_artifacts(artifacts_to_retrieve)


def _extract_archives_into_subdirectories(output_dir: Path):
    """
    Extracts all files in output_dir to output_dir/{filename}, matching
    the directory structure of the `therock-archives` CMake target. This
    operation is different from the modes in other files that merge the
    extracted files by component type or flatten into just bin/dist/.
    """
    # TODO(scotttodd): Move this code to artifacts.py? should move more of this
    #                  logic into that file and add comprehensive unit tests
    log(f"Extracting archives in '{output_dir}'")
    archive_files = list(output_dir.glob("*.tar.*"))
    for archive_file in archive_files:
        # Get (for example) 'amd-llvm_lib_generic' from '/path/to/amd-llvm_lib_generic.tar.xz'
        # We can't just use .stem since that only removes the last extension.
        #   1. .name gets us 'amd-llvm_lib_generic.tar.xz'
        #   2. .partition('.') gets (before, sep, after), discard all but 'before'
        archive_file_stem, _, _ = archive_file.name.partition(".")

        with tarfile.TarFile.open(archive_file, mode="r:xz") as tf:
            log(f"++ Extracting '{archive_file.name}' to '{archive_file_stem}'")
            tf.extractall(output_dir / archive_file_stem, filter="tar")


def run(args):
    run_id = args.run_id
    target = args.target
    output_dir = args.output_dir
    if not output_dir.is_dir():
        log(f"Output dir '{output_dir}' does not exist. Exiting...")
        return
    s3_artifacts = retrieve_s3_artifacts(run_id, target)
    if not s3_artifacts:
        log(f"S3 artifacts for {run_id} does not exist. Exiting...")
        return

    if args.all:
        retrieve_all_artifacts(run_id, target, output_dir, s3_artifacts)
    else:
        retrieve_base_artifacts(args, run_id, output_dir, s3_artifacts)
        if not args.base_only:
            retrieve_enabled_artifacts(args, target, run_id, output_dir, s3_artifacts)

    if args.extract:
        _extract_archives_into_subdirectories(output_dir)


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
        "--output-dir",
        type=Path,
        default="build/artifacts",
        help="Output path for fetched artifacts, defaults to `build/artifacts/` as in source builds",
    )

    parser.add_argument(
        "--extract",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Extract files after fetching them",
    )

    artifacts_group = parser.add_argument_group("artifacts_group")
    artifacts_group.add_argument(
        "--all",
        default=False,
        help="Include all artifacts",
        action=argparse.BooleanOptionalAction,
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

    if args.all and (
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
