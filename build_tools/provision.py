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
import urllib.request
from pathlib import Path
import requests
import tarfile
from tqdm import tqdm


def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def url_exists(url):
    try:
        response = urllib.request.urlopen(url)
        return response.status == 200
    except urllib.error.URLError:
        return False


def _untar_files(output_dir):
    output_dir_path = Path(output_dir).resolve()
    tar_file_paths = output_dir_path.glob("*.tar.*")
    for file_path in tar_file_paths:
        with tarfile.open(file_path) as extracted_tar_file:
            extracted_tar_file.extractall(output_dir)
        file_path.unlink()


def create_output_directory(args):
    output_dir_path = args.output_dir
    log(f"Creating directory {output_dir_path}")
    if os.path.isdir(output_dir_path):
        log(
            f"Directory {output_dir_path} already exists, removing existing directory and files"
        )
        shutil.rmtree(output_dir_path)
    os.mkdir(output_dir_path)
    log(f"Created directory {output_dir_path}")


def get_github_release_assets(release_id, amdgpu_family):
    github_release_url = (
        f"https://api.github.com/repos/ROCm/TheRock/releases/tags/{release_id}"
    )
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    response = requests.get(github_release_url, headers=headers)
    if response.status_code != 200:
        print(
            f"Error when retrieving GitHub release assets for release ID {release_id}. Exiting..."
        )
        return

    release_data = response.json()
    # If this release was "nightly-release" or "dev-release", we sort to retrieve the most recent release asset
    asset_data = sorted(
        release_data["assets"], key=lambda item: item["updated_at"], reverse=True
    )
    for asset in asset_data:
        if amdgpu_family in asset["name"]:
            return asset
    return None


def download_github_release_asset(asset_data, output_dir):
    asset_name = asset_data["name"]
    asset_url = asset_data["url"]
    destination = Path(output_dir) / asset_name
    headers = {"Accept": "application/octet-stream"}
    response = requests.get(asset_url, stream=True, headers=headers)

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(destination, "wb") as file:
            for chunk in response.iter_content(block_size * block_size):
                progress_bar.update(len(chunk))
                file.write(chunk)

    _untar_files(output_dir)


def retrieve_artifacts_by_ci(args):
    runner_id = args.runner_id
    output_dir = args.output_dir
    amdgpu_family = args.amdgpu_family
    if not s3_bucket_exists(runner_id):
        print(f"S3 artifacts for {runner_id} does not exist. Exiting...")
        return

    args.all = True
    print(f"Retrieving artifacts for runner ID {runner_id}")
    retrieve_base_artifacts(args, runner_id, output_dir)
    retrieve_enabled_artifacts(args, True, amdgpu_family, runner_id, output_dir)
    _untar_files(output_dir)
    print(f"Retrieved artifacts for runner ID {runner_id}")


def retrieve_artifacts_by_release(args):
    release_id = args.release_id
    output_dir = args.output_dir
    amdgpu_family = args.amdgpu_family
    print(f"Retrieving artifacts for release ID {release_id}")
    asset_data = get_github_release_assets(release_id, amdgpu_family)
    if not asset_data:
        print(f"GitHub release asset for {release_id} not found. Exiting...")
        return
    download_github_release_asset(asset_data, output_dir)
    print(f"Retrieving artifacts for runner ID {release_id}")


def run(args):
    create_output_directory(args)
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
        help="AMD GPU family to provision (please refer to this: https://github.com/ROCm/TheRock/blob/main/cmake/therock_amdgpu_targets.cmake#L44-L81 for target choices)",
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
