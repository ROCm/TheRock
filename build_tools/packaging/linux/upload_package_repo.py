#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


"""Given ROCm artifacts directories, performs packaging to
create RPM and DEB packages and stores to OUTPUT folder

```
python ./build_tools/packaging/linux/upload_package_repo.py \
             --pkg-type deb \
             --s3-bucket therock-deb-rpm-test \
             --amdgpu-family gfx94X-dcgpu \
             --artifact-id 16418185899
```
"""

import os
import argparse
import subprocess
import boto3
import shutil
import datetime


def run_command(cmd, cwd=None):
    """
    Function to execute commands in shell.
    """
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def find_package_dir():
    """
    Finds the default output dir for packages.
    Expects packages in ./output/packages
    """
    base_dir = os.path.join(os.getcwd(), "output", "packages")
    if not os.path.exists(base_dir):
        raise RuntimeError(f"Package directory not found: {base_dir}")
    print(f"Using package directory: {base_dir}")
    return base_dir


def create_deb_repo(package_dir, origin_name):
    """Function to create rpm repo
    It takes all the rpm files in the package_dir parameter
    And creates the deb package list using dpkg-scanpackages
    Package list is gzipped Packages.gz to pool/main foldre
    Also create Release meta package file needed for debian repo

    Parameters:
    package_dir : Folder to search for deb packages
    origin_name : S3 bucket to upload, used in meta data creation

    Returns: None
    """
    print("Creating APT repository...")
    dists_dir = os.path.join(package_dir, "dists", "stable", "main", "binary-amd64")
    release_dir = os.path.join(package_dir, "dists", "stable")
    pool_dir = os.path.join(package_dir, "pool", "main")

    os.makedirs(dists_dir, exist_ok=True)
    os.makedirs(pool_dir, exist_ok=True)
    for file in os.listdir(package_dir):
        if file.endswith(".deb"):
            shutil.move(os.path.join(package_dir, file), os.path.join(pool_dir, file))

    print(
        "Generating Packages file at repository root so 'Filename' paths are 'pool/...'."
    )
    cmd = "dpkg-scanpackages -m pool/main /dev/null > dists/stable/main/binary-amd64/Packages"
    run_command(cmd, cwd=package_dir)
    run_command("gzip -9c Packages > Packages.gz", cwd=dists_dir)

    print("Creating Release file...")
    release_content = f"""\
Origin: {origin_name}
Label: {origin_name}
Suite: stable
Codename: stable
Version: 1.0
Date: {datetime.datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S UTC')}
Architectures: amd64
Components: main
Description: ROCm Repository
"""
    os.makedirs(release_dir, exist_ok=True)
    release_path = os.path.join(release_dir, "Release")
    with open(release_path, "w") as f:
        f.write(release_content)
    print(f"Wrote Release file to {release_path}")


def create_rpm_repo(package_dir):
    """Function to create rpm repo
    It takes all the rpm files in the package_dir parameter
    And creates the rpm repo using createrepo_c command inside x86_64 folder

    Parameters:
    package_dir : Folder to search for rpm packages

    Returns: None
    """
    print("Creating YUM/DNF repository...")

    arch_dir = os.path.join(package_dir, "x86_64")
    os.makedirs(arch_dir, exist_ok=True)
    for file in os.listdir(package_dir):
        if file.endswith(".rpm"):
            shutil.move(os.path.join(package_dir, file), os.path.join(arch_dir, file))
    run_command("createrepo_c .", cwd=arch_dir)
    print(f"Generated repodata/ in {arch_dir}")


def upload_to_s3(source_dir, bucket, prefix):
    """Function to upload the packges and repo files to the s3 bucket
    It upload the source_dir contents to s3://{bucket}/{prefix}/

    Parameters:
    source_dir : Folder with the packages and repo files
    bucket : S3 bucket
    prefix : S3 prefix

    Returns: None
    """
    s3 = boto3.client("s3")
    print(f"Uploading to s3://{bucket}/{prefix}/")

    for root, _, files in os.walk(source_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            rel_path = os.path.relpath(local_path, source_dir)
            s3_key = os.path.join(prefix, rel_path).replace("\\", "/")
            print(f"Uploading: {local_path} → s3://{bucket}/{s3_key}")
            s3.upload_file(local_path, bucket, s3_key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pkg-type",
        required=True,
        choices=["deb", "rpm"],
        help="Type of packages to process",
    )
    parser.add_argument("--s3-bucket", required=True, help="Target S3 bucket name")
    parser.add_argument(
        "--amdgpu-family", required=True, help="AMDGPU family identifier (e.g., gfx94X)"
    )
    parser.add_argument(
        "--artifact-id", required=True, help="Unique artifact ID or version tag"
    )
    args = parser.parse_args()

    package_dir = find_package_dir()
    s3_prefix = f"{args.amdgpu_family}_{args.artifact_id}/{args.pkg_type}"

    if args.pkg_type == "deb":
        create_deb_repo(package_dir, args.s3_bucket)
    else:
        create_rpm_repo(package_dir)

    upload_to_s3(package_dir, args.s3_bucket, s3_prefix)


if __name__ == "__main__":
    main()
