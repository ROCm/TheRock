#!/usr/bin/env python3

import os
import argparse
import subprocess
import boto3
import shutil

def run_command(cmd, cwd=None):
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

def create_deb_repo(package_dir):
    print("Creating APT repository...")
    dists_dir = os.path.join(package_dir, "dists")
    pool_dir = os.path.join(package_dir, "pool")
    os.makedirs(dists_dir, exist_ok=True)
    os.makedirs(pool_dir, exist_ok=True)

    for file in os.listdir(package_dir):
        if file.endswith(".deb"):
            shutil.move(os.path.join(package_dir, file), pool_dir)

    run_command("dpkg-scanpackages pool /dev/null | gzip -9c > dists/Packages.gz", cwd=package_dir)

def create_rpm_repo(package_dir):
    print("Creating YUM repository...")
    run_command("createrepo .", cwd=package_dir)

def upload_to_s3(source_dir, bucket, prefix):
    s3 = boto3.client("s3")
    print(f"Uploading to s3://{bucket}/{prefix}/")
    for root, _, files in os.walk(source_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            s3_key = os.path.join(prefix, os.path.relpath(local_path, source_dir))
            print(f"Uploading: {local_path} → s3://{bucket}/{s3_key}")
            s3.upload_file(local_path, bucket, s3_key)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkg-type", required=True, choices=["deb", "rpm"])
    parser.add_argument("--s3-bucket", required=True)
    parser.add_argument("--amdgpu-family", required=True)
    parser.add_argument("--artifact-id", required=True)
    args = parser.parse_args()

    package_dir = find_package_dir()
    s3_prefix = f"{args.amdgpu_family}_{args.artifact_id}/{args.pkg_type}"

    if args.pkg_type == "deb":
        create_deb_repo(package_dir)
    else:
        create_rpm_repo(package_dir)

    upload_to_s3(package_dir, args.s3_bucket, s3_prefix)

if __name__ == "__main__":
    main()
