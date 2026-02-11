#!/usr/bin/env python

"""Fetches artifacts from S3.

The install_rocm_from_artifacts.py script builds on top of this script to both
download artifacts then unpack them into a usable install directory.

Example usage (using https://github.com/ROCm/TheRock/actions/runs/15685736080):
  pip install boto3
  python build_tools/fetch_artifacts.py \
    --run-id 15685736080 --artifact-group gfx110X-all --output-dir ~/.therock/artifacts_15685736080

Include/exclude regular expressions can be given to control what is downloaded:
  python build_tools/fetch_artifacts.py \
    --run-id 15685736080 --artifact-group gfx110X-all --output-dir ~/.therock/artifacts_15685736080 \
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
import concurrent.futures
from pathlib import Path
import platform
import re
import shutil
import sys

from _therock_utils.artifact_backend import ArtifactBackend, S3Backend
from _therock_utils.artifacts import (
    ArtifactPopulator,
    _open_archive_for_read,
)
from artifact_manager import DownloadRequest, download_artifact
from github_actions.github_actions_utils import retrieve_bucket_info


# TODO(geomin12): switch out logging library
def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def list_artifacts_for_group(backend: ArtifactBackend, artifact_group: str) -> set[str]:
    """Lists artifacts from backend, filtered by artifact_group.

    Args:
        backend: ArtifactBackend instance configured for the target run
        artifact_group: GPU family to filter by (e.g., "gfx94X-all"). Also includes
            artifacts with "generic" in the name.

    Returns:
        Set of artifact filenames matching the artifact_group or "generic".
    """
    log(f"Retrieving artifacts from '{backend.base_uri}'")

    # Get all artifacts from backend
    all_artifacts = backend.list_artifacts()

    # Filter by artifact_group (matches if artifact_group or "generic" in filename)
    data = set()
    for filename in all_artifacts:
        if artifact_group in filename or "generic" in filename:
            data.add(filename)

    if not data:
        log(f"Found no artifacts matching '{artifact_group}' at '{backend.base_uri}'")
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


def get_postprocess_mode(args) -> str | None:
    """Returns 'extract', 'flatten' or None (default is 'extract')."""
    if args.flatten:
        return "flatten"
    if args.no_extract:
        return None
    return "extract"


def extract_artifact(
    archive_path: Path, *, delete_archive: bool, postprocess_mode: str
):
    """Extracts and postprocesses an artifact from an archive file in-place.

    Args:
        archive_path: Path to the archive (e.g. `amd-llvm_lib_generic.tar.xz`)
        delete_archive: True to delete the archive after extraction
        postprocess_mode: Either 'flatten' or 'extract'
          * 'flatten' merges artifacts into a single "dist/" directory
          * 'extract' puts each artifact in a dir (e.g. `amd-llvm_lib_generic/`)
    """
    # Get (for example) 'amd-llvm_lib_generic' from '/path/to/amd-llvm_lib_generic.tar.xz'
    # We can't just use .stem since that only removes the last extension.
    #   1. .name gets us 'amd-llvm_lib_generic.tar.xz'
    #   2. .partition('.') gets (before, sep, after), discard all but 'before'
    archive_file = archive_path
    artifact_name, *_ = archive_file.name.partition(".")

    if postprocess_mode == "extract":
        output_dir = archive_file.parent / artifact_name
        if output_dir.exists():
            shutil.rmtree(output_dir)
        with _open_archive_for_read(archive_file) as tf:
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
    artifact_group = args.artifact_group
    output_dir = args.output_dir

    external_repo, bucket = retrieve_bucket_info(
        github_repository=run_github_repo,
        workflow_run_id=run_id,
    )
    backend = S3Backend(
        bucket=bucket,
        run_id=run_id,
        platform=args.platform,
        external_repo=external_repo,
    )

    # Lookup which artifacts exist in the bucket.
    # Note: this currently does not check that all requested artifacts
    # (via include patterns) do exist, so this may silently fail to fetch
    # expected files.
    available_artifacts = list_artifacts_for_group(
        backend=backend, artifact_group=artifact_group
    )
    if not available_artifacts:
        log(f"No matching artifacts for {run_id} exist. Exiting...")
        sys.exit(1)

    # Include/exclude filtering.
    filtered_artifacts = filter_artifacts(
        available_artifacts, args.include, args.exclude
    )
    if not filtered_artifacts:
        log(f"Filtering artifacts for {run_id} resulted in an empty set. Exiting...")
        sys.exit(1)

    download_requests = [
        DownloadRequest(
            artifact_key=artifact,
            dest_path=output_dir / artifact,
            backend=backend,
        )
        for artifact in sorted(filtered_artifacts)
    ]

    download_summary = "\n  ".join(
        [f"{req.backend.base_uri}/{req.artifact_key}" for req in download_requests]
    )
    log(f"\nFiltered artifacts to download:\n  {download_summary}\n")

    if args.dry_run:
        log("Skipping downloads since --dry-run was set")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Download and extract in parallel.
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.download_concurrency
    ) as download_executor:
        download_futures = [
            download_executor.submit(download_artifact, req)
            for req in download_requests
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
                # download_artifact returns Optional[Path] - None on failure
                download_result = download_future.result(timeout=60)
                if download_result is None:
                    continue
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
        "--artifact-group",
        type=str,
        required=True,
        help="Artifact group to fetch",
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
        help="GitHub repository for --run-id in 'owner/repo' format (e.g. 'ROCm/TheRock'). Defaults to GITHUB_REPOSITORY env var or 'ROCm/TheRock'",
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
