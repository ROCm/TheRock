#!/usr/bin/env python
"""fetch_artifacts.py

This script fetches artifacts from s3. See also docs/development/artifacts.md.

The install_rocm_from_artifacts.py script builds on top of this script to both
download artifacts then unpack them into a usable install directory.

Note: This script currently only retrieves the requested artifacts, it does not
model inter-artifact dependencies.
"""

import argparse
import concurrent.futures
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
import platform
from shutil import copyfileobj
import sys
import urllib.request

from _therock_utils.artifacts import ArtifactName

THEROCK_DIR = Path(__file__).resolve().parent.parent

# Importing build_artifact_upload.py
sys.path.append(str(THEROCK_DIR / "build_tools" / "github_actions"))
from upload_build_artifacts import retrieve_bucket_info

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


@dataclass
class ArtifactDownloadRequest:
    """Information about a request to download an artifact to a local path."""

    artifact_url: str
    output_path: Path
    dry_run: bool


def retrieve_s3_artifacts(run_id, amdgpu_family) -> set[str]:
    """Checks that the AWS S3 bucket exists and returns artifact file names."""
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
            artifact_names = set()
            for file_name in parser.files:
                # We only want to get .tar.xz files, not .tar.xz.sha256sum
                if "sha256sum" not in file_name and "tar.xz" in file_name:
                    artifact_names.add(file_name)
            return artifact_names
    except urllib.request.HTTPError as err:
        if err.code == 404:
            raise ArtifactNotFoundExeption(
                f"No artifacts found for {run_id}-{PLATFORM} with amdgpu_family {amdgpu_family}"
            )
        else:
            raise FetchArtifactException(
                f"Error when retrieving S3 bucket {run_id}-{PLATFORM}/index-{amdgpu_family}.html"
            )


def collect_artifacts_download_requests(
    artifact_names: list[str],
    run_id: str,
    output_dir: Path,
    dry_run: bool = False,
) -> list[str]:
    """Collects S3 artifact URLs to execute later in parallel."""
    EXTERNAL_REPO, BUCKET = retrieve_bucket_info()
    BUCKET_URL = f"https://{BUCKET}.s3.us-east-2.amazonaws.com/{EXTERNAL_REPO}{run_id}-{PLATFORM}"
    artifacts_to_retrieve = []
    for artifact_name in artifact_names:
        artifacts_to_retrieve.append(
            ArtifactDownloadRequest(
                artifact_url=f"{BUCKET_URL}/{artifact_name}",
                output_path=output_dir / artifact_name,
                dry_run=dry_run,
            )
        )

    return artifacts_to_retrieve


def download_artifact(artifact_download_request: ArtifactDownloadRequest):
    if artifact_download_request.dry_run:
        log(
            f"++ (Dry-run) Would download from {artifact_download_request.artifact_url} to {artifact_download_request.output_path}"
        )
        return

    log(
        f"++ Downloading from {artifact_download_request.artifact_url} to {artifact_download_request.output_path}"
    )
    with urllib.request.urlopen(
        artifact_download_request.artifact_url
    ) as in_stream, open(artifact_download_request.output_path, "wb") as out_file:
        copyfileobj(in_stream, out_file)
    log(f"++ Download complete for {artifact_download_request.output_path}")


def download_artifacts(artifact_download_requests: list[ArtifactDownloadRequest]):
    """Downloads artifacts in parallel using a thread pool executor."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(download_artifact, artifact_download_request)
            for artifact_download_request in artifact_download_requests
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result(timeout=60)


def compute_enabled_artifacts(
    s3_artifacts: set[str],
    names: list[str] | None,
    generic_only: bool,
    target: str,
    dev: bool,
    lib: bool,
    test: bool,
    verbose: bool = False,
):
    """Determines enabled artifacts based on arguments."""

    if verbose:
        log(f"Considering sartifacts found on S3:")

    enabled_artifact_names = []
    for artifact in sorted(list(s3_artifacts)):
        an = ArtifactName.from_filename(artifact)
        if verbose:
            log(f"  {artifact} ({an.name}, {an.component}, {an.target_family})")

        if names and an.name not in names:
            continue

        if an.target_family != "generic":
            if generic_only:
                continue
            if target != an.target_family:
                continue

        if an.component == "dev":
            if dev:
                enabled_artifact_names.append(artifact)
        elif an.component == "lib":
            if lib:
                enabled_artifact_names.append(artifact)
        elif an.component == "test":
            if test:
                enabled_artifact_names.append(artifact)
        else:
            continue  # skip

    if verbose:
        log("Enabled artifacts:")
        for artifact in enabled_artifact_names:
            log(f"  {artifact} ({an.name}, {an.component}, {an.target_family})")

    return enabled_artifact_names


def retrieve_enabled_artifacts(
    args: argparse.Namespace,
    target: str,
    run_id: str,
    output_dir: Path,
    s3_artifacts: set[str],
    verbose: bool = False,
    dry_run: bool = False,
):
    """Retrieves artifacts using urllib, based on the enabled arguments."""

    enabled_artifact_names = compute_enabled_artifacts(
        s3_artifacts=s3_artifacts,
        names=None if not args.names else args.names.split(","),
        generic_only=args.generic_only,
        target=target,
        dev=args.dev,
        lib=args.lib,
        test=args.test,
        verbose=verbose,
    )
    artifacts_to_retrieve = collect_artifacts_download_requests(
        artifact_names=enabled_artifact_names,
        run_id=run_id,
        output_dir=output_dir,
        dry_run=dry_run,
    )
    download_artifacts(artifacts_to_retrieve)


def run(args: argparse.Namespace):
    run_id = args.run_id
    target = args.target
    output_dir = args.output_dir

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=False)

    s3_artifacts = retrieve_s3_artifacts(run_id, target)
    if not s3_artifacts:
        print(f"S3 artifacts for '{run_id}' do not exist. Exiting...")
        return

    retrieve_enabled_artifacts(
        args=args,
        target=target,
        run_id=run_id,
        output_dir=output_dir,
        s3_artifacts=s3_artifacts,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )


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
        "--generic-only", help="Include only generic artifacts", action="store_true"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=THEROCK_DIR / "build" / "artifacts",
        help="Path to the artifact output directory (e.g. build/artifacts)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose information",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which files would be fetched without actually downloading",
    )

    parser.add_argument(
        "--names",
        default="",
        help="Comma-delimited list of artifact names to fetch (e.g. 'prim,rand') or omit to fetch all",
    )

    components_group = parser.add_argument_group("Components")
    components_group.add_argument(
        "--dev",
        default=False,
        help="Include 'dev' artifacts (default off)",
        action=argparse.BooleanOptionalAction,
    )
    components_group.add_argument(
        "--lib",
        default=True,
        help="Include 'lib' artifacts (default on)",
        action=argparse.BooleanOptionalAction,
    )
    components_group.add_argument(
        "--test",
        default=False,
        help="Include 'test' artifacts (default off)",
        action=argparse.BooleanOptionalAction,
    )
    # TODO: also include doc and run?

    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
