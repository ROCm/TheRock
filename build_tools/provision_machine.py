#!/usr/bin/env python
"""provision_machine.py

This script helps CI workflows, developers and testing suites easily provision TheRock to their environment.
It provisions TheRock to an output directory from one of these sources:
- GitHub CI workflow run
- GitHub release tag
- An existing installation of TheRock

Usage:
python build_tools/provision_machine.py [--output-dir OUTPUT_DIR] [--amdgpu-family AMDGPU_FAMILY] (--run-id RUN_ID | --release RELEASE | --input-dir INPUT_DIR)

Examples:
- Downloads the gfx94X S3 artifacts from GitHub CI workflow run 14474448215 (from https://github.com/ROCm/TheRock/actions/runs/14474448215) to the default output directory `therock-build`:
    - `python build_tools/provision_machine.py --run-id 14474448215 --amdgpu-family gfx94X-dcgpu`
- Downloads the latest gfx110X artifacts from GitHub release tag `nightly-release` to the specified output directory `build`:
    - `python build_tools/provision_machine.py --release latest --amdgpu-family gfx110X-dgpu --output-dir build`
- Downloads the version `6.4.0rc20250416` gfx110X artifacts from GitHub release tag `nightly-release` to the specified output directory `build`:
    - `python build_tools/provision_machine.py --release 6.4.0rc20250416 --amdgpu-family gfx110X-dgpu --output-dir build`
- Downloads the version `6.4.0.dev0+8f6cdfc0d95845f4ca5a46de59d58894972a29a9` gfx120X artifacts from GitHub release tag `dev-release` to the default output directory `therock-build`:
    - `python build_tools/provision_machine.py --release 6.4.0.dev0+8f6cdfc0d95845f4ca5a46de59d58894972a29a9 --amdgpu-family gfx120X-all`

You can select your AMD GPU family from this file https://github.com/ROCm/TheRock/blob/59c324a759e8ccdfe5a56e0ebe72a13ffbc04c1f/cmake/therock_amdgpu_targets.cmake#L44-L81

Note: the script will overwrite the output directory argument. If no argument is passed, it will overwrite the default "therock-build" directory.
"""

import argparse
import sys
import os
import shutil
from fetch_artifacts import (
    retrieve_base_artifacts,
    retrieve_enabled_artifacts,
    s3_bucket_exists,
)
from pathlib import Path
import tarfile
from tqdm import tqdm
from _therock_utils.artifacts import ArtifactPopulator
import requests
from packaging.version import Version, InvalidVersion


def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def _untar_files(output_dir, destination):
    """
    Retrieves all tar files in the output_dir, then extracts all files to the output_dir
    """
    output_dir_path = Path(output_dir).resolve()
    # In order to get better visibility on untar-ing status, tqdm adds a progress bar
    log(f"Extracting {destination.name} to {output_dir}")
    with tarfile.open(destination) as extracted_tar_file:
        for member in tqdm(
            iterable=extracted_tar_file.getmembers(),
            total=len(extracted_tar_file.getmembers()),
        ):
            extracted_tar_file.extract(member=member, path=output_dir_path)
    destination.unlink()


def _create_output_directory(args):
    """
    If the output directory already exists, delete it and its contents.
    Then, create the output directory.
    """
    output_dir_path = args.output_dir
    log(f"Creating directory {output_dir_path}")
    if os.path.isdir(output_dir_path):
        log(
            f"Directory {output_dir_path} already exists, removing existing directory and files"
        )
        shutil.rmtree(output_dir_path)
    os.mkdir(output_dir_path)
    log(f"Created directory {output_dir_path}")


def _get_github_release_assets(release_tag, amdgpu_family, release_version):
    """
    Makes an API call to retrieve the release's assets, then retrieves the asset matching the amdgpu family
    """
    github_release_url = (
        f"https://api.github.com/repos/ROCm/TheRock/releases/tags/{release_tag}"
    )
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    # If GITHUB_TOKEN environment variable is available, include it in the API request to avoid a lower rate limit
    gh_token = os.getenv("GITHUB_TOKEN", "")
    if gh_token:
        headers["Authentication"] = f"Bearer {gh_token}"

    response = requests.get(github_release_url, headers=headers)
    if response.status_code == 403:
        log(
            f"Error when retrieving GitHub release assets for release tag '{release_tag}'. This is most likely a rate limiting issue, so please try again"
        )
        return
    elif response.status_code != 200:
        log(
            f"Error when retrieving GitHub release assets for release tag '{release_tag}'. Exiting..."
        )
        return

    release_data = response.json()

    # We retrieve the most recent release asset that matches the amdgpu_family
    # In the cases of "nightly-release" or "dev-release", this will retrieve the the specified version or latest
    asset_data = sorted(
        release_data["assets"], key=lambda item: item["updated_at"], reverse=True
    )

    # For nightly-release
    if release_tag == "nightly-release" and release_version != "latest":
        for asset in asset_data:
            if amdgpu_family in asset["name"] and release_version in asset["name"]:
                return asset
    # For dev-release
    elif release_tag == "dev-release":
        for asset in asset_data:
            if amdgpu_family in asset["name"] and release_version in asset["name"]:
                return asset
    # Otherwise, return the latest and amdgpu-matched asset available from the tag nightly-release
    elif release_tag == "nightly-release" and release_version == "latest":
        for asset in asset_data:
            if amdgpu_family in asset["name"]:
                return asset

    return None


def _download_github_release_asset(asset_data, output_dir):
    """
    With the GitHub asset data, this function downloads the asset to the output_dir
    """
    asset_name = asset_data["name"]
    asset_url = asset_data["url"]
    destination = Path(output_dir) / asset_name
    headers = {"Accept": "application/octet-stream"}
    # Making the API call to retrieve the asset
    response = requests.get(asset_url, stream=True, headers=headers)

    # Downloading the asset in chunks to destination
    # In order to get better visibility on downloading status, tqdm adds a progress bar
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(destination, "wb") as file:
            for chunk in response.iter_content(block_size * block_size):
                progress_bar.update(len(chunk))
                file.write(chunk)

    # After downloading the asset, untar-ing the file
    _untar_files(output_dir, destination)


def retrieve_artifacts_by_run_id(args):
    """
    If the user requested TheRock artifacts by CI (run ID), this function will retrieve those assets
    """
    run_id = args.run_id
    output_dir = args.output_dir
    amdgpu_family = args.amdgpu_family
    log(f"Retrieving artifacts for run ID {run_id}")
    if not s3_bucket_exists(run_id):
        log(f"S3 artifacts for {run_id} does not exist. Exiting...")
        return

    # Retrieving base and all math-lib tar artifacts and downloading them to output_dir
    retrieve_base_artifacts(args, run_id, output_dir)
    retrieve_enabled_artifacts(args, amdgpu_family, run_id, output_dir)

    # Flattening artifacts from .tar* files then removing .tar* files
    log(f"Untar-ing artifacts for {run_id}")
    output_dir_path = Path(output_dir).resolve()
    tar_file_paths = list(output_dir_path.glob("*.tar.*"))
    flattener = ArtifactPopulator(
        output_path=output_dir_path, verbose=True, flatten=True
    )
    flattener(*tar_file_paths)
    for file_path in tar_file_paths:
        file_path.unlink()

    log(f"Retrieved artifacts for run ID {run_id}")


def retrieve_artifacts_by_release(args):
    """
    If the user requested TheRock artifacts by release version, this function will retrieve those assets
    """
    output_dir = args.output_dir
    amdgpu_family = args.amdgpu_family
    # In the case that the user passes in latest, we will get the latest nightly-release
    if args.release == "latest":
        release_tag = "nightly-release"
    # Otherwise, determine if version is nightly-release or dev-release
    else:
        try:
            version = Version(args.release)
            if not version.is_devrelease and not version.is_prerelease:
                log(f"This script requires a nightly-release or dev-release version.")
                log("Please retrieve the correct release version from:")
                log(
                    "\t - https://github.com/ROCm/TheRock/releases/tag/nightly-release (nightly-release example: 6.4.0rc20250416)"
                )
                log(
                    "\t - https://github.com/ROCm/TheRock/releases/tag/dev-release (dev-release example: 6.4.0.dev0+8f6cdfc0d95845f4ca5a46de59d58894972a29a9)"
                )
                log("Exiting...")
                return
            release_tag = "dev-release" if version.is_devrelease else "nightly-release"
        except InvalidVersion:
            log(f"Invalid release version '{args.release}'")
            log("Please retrieve the correct release version from:")
            log(
                "\t - https://github.com/ROCm/TheRock/releases/tag/nightly-release (nightly-release example: 6.4.0rc20250416)"
            )
            log(
                "\t - https://github.com/ROCm/TheRock/releases/tag/dev-release (dev-release example: 6.4.0.dev0+8f6cdfc0d95845f4ca5a46de59d58894972a29a9)"
            )
            log("Exiting...")
            return
    release_version = args.release

    log(f"Retrieving artifacts for release tag {release_tag}")
    asset_data = _get_github_release_assets(release_tag, amdgpu_family, release_version)
    if not asset_data:
        log(f"GitHub release asset for '{release_tag}' not found. Exiting...")
        return
    _download_github_release_asset(asset_data, output_dir)
    log(f"Retrieving artifacts for run ID {release_tag}")


def run(args):
    log("### Provisioning TheRock 🪨 ###")
    _create_output_directory(args)
    if args.run_id:
        retrieve_artifacts_by_run_id(args)
    elif args.release:
        retrieve_artifacts_by_release(args)


def main(argv):
    parser = argparse.ArgumentParser(prog="provision")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./therock-build",
        help="Path of the output directory for TheRock",
    )

    parser.add_argument(
        "--amdgpu-family",
        type=str,
        default="gfx94X-dcgpu",
        help="AMD GPU family to provision (please refer to this: https://github.com/ROCm/TheRock/blob/59c324a759e8ccdfe5a56e0ebe72a13ffbc04c1f/cmake/therock_amdgpu_targets.cmake#L44-L81 for family choices)",
    )

    # This mutually exclusive group will ensure that only one argument is present
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run-id", type=str, help="GitHub run ID of TheRock to provision"
    )

    group.add_argument(
        "--release",
        type=str,
        help="Github release version of TheRock to provision, from the nightly-release (X.Y.ZrcYYYYMMDD) or dev-release (X.Y.Z.dev0+{hash})",
    )

    artifacts_group = parser.add_argument_group("artifacts_group")
    artifacts_group.add_argument(
        "--blas",
        default=True,
        help="If flagged, blas artifacts will be retrieved",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--fft",
        default=True,
        help="If flagged, fft artifacts will be retrieved",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--miopen",
        default=True,
        help="If flagged, miopen artifacts will be retrieved",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--prim",
        default=True,
        help="If flagged, prim artifacts will be retrieved",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--rand",
        default=True,
        help="If flagged, rand artifacts will be retrieved",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--rccl",
        default=True,
        help="If flagged, rccl artifacts will be retrieved",
        action=argparse.BooleanOptionalAction,
    )

    artifacts_group.add_argument(
        "--test",
        default=True,
        help="If flagged, test artifacts will be retrieved",
        action=argparse.BooleanOptionalAction,
    )

    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
