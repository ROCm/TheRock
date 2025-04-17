#!/usr/bin/env python

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


def _get_github_release_assets(release_id, amdgpu_family):
    """
    Makes an API call to retrieve the release's assets, then retrieves the asset matching the amdgpu family
    """
    github_release_url = (
        f"https://api.github.com/repos/ROCm/TheRock/releases/tags/{release_id}"
    )
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    response = requests.get(github_release_url, headers=headers)
    if response.status_code == 403:
        log(
            f"Error when retrieving GitHub release assets for release ID {release_id}. This is most likely a rate limiting issue, so please try again"
        )
        return
    elif response.status_code != 200:
        log(
            f"Error when retrieving GitHub release assets for release ID {release_id}. Exiting..."
        )
        return

    release_data = response.json()

    # We retrieve the most recent release asset that matches the amdgpu_family
    # In the cases of "nightly-release" or "dev-release", this will retrieve the most recent release asset
    asset_data = sorted(
        release_data["assets"], key=lambda item: item["updated_at"], reverse=True
    )
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


def retrieve_artifacts_by_ci(args):
    """
    If the user requested TheRock artifacts by CI (runner ID), this function will retrieve those assets
    """
    runner_id = args.runner_id
    output_dir = args.output_dir
    amdgpu_family = args.amdgpu_family
    log(f"Retrieving artifacts for runner ID {runner_id}")
    if not s3_bucket_exists(runner_id):
        log(f"S3 artifacts for {runner_id} does not exist. Exiting...")
        return

    args.all = True

    # Retrieving base and all math-lib tar artifacts and downloading them to output_dir
    retrieve_base_artifacts(args, runner_id, output_dir)
    retrieve_enabled_artifacts(args, True, amdgpu_family, runner_id, output_dir)

    # Flattening artifacts from .tar* files then removing .tar* files
    log(f"Untar-ing artifacts for {runner_id}")
    output_dir_path = Path(output_dir).resolve()
    tar_file_paths = list(output_dir_path.glob("*.tar.*"))
    flattener = ArtifactPopulator(
        output_path=output_dir_path, verbose=True, flatten=True
    )
    flattener(*tar_file_paths)
    for file_path in tar_file_paths:
        file_path.unlink()

    log(f"Retrieved artifacts for runner ID {runner_id}")


def retrieve_artifacts_by_release(args):
    """
    If the user requested TheRock artifacts by release tag (release ID), this function will retrieve those assets
    """
    release_id = args.release_id
    output_dir = args.output_dir
    amdgpu_family = args.amdgpu_family
    log(f"Retrieving artifacts for release ID {release_id}")
    asset_data = _get_github_release_assets(release_id, amdgpu_family)
    if not asset_data:
        log(f"GitHub release asset for {release_id} not found. Exiting...")
        return
    _download_github_release_asset(asset_data, output_dir)
    log(f"Retrieving artifacts for runner ID {release_id}")


def run(args):
    log("### Provisioning TheRock 🪨 ###")
    _create_output_directory(args)
    if args.runner_id:
        retrieve_artifacts_by_ci(args)

    if args.release_id:
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
        "--runner-id", type=str, help="GitHub runner ID of TheRock to provision"
    )

    group.add_argument(
        "--release-id", type=str, help="Github release ID of TheRock to provision"
    )

    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
