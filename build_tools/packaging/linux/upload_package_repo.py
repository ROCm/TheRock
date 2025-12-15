#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Packaging + repository upload tool.

Nightly mode:
  /deb/<mmddyy>
  /rpm/<mmddyy>
"""

import os
import argparse
import subprocess
import boto3
import shutil
import datetime

SVG_DEFS = """<svg xmlns="http://www.w3.org/2000/svg" style="display:none">
<defs>
  <symbol id="file" viewBox="0 0 265 323">
    <path fill="#4582ec" d="M213 115v167a41 41 0 01-41 41H69a41 41 0 01-41-41V39a39 39 0 0139-39h127a39 39 0 0139 39v76z"/>
    <path fill="#77a4ff" d="M176 17v88a19 19 0 0019 19h88"/>
  </symbol>
  <symbol id="folder-shortcut" viewBox="0 0 265 216">
    <path fill="#4582ec" d="M18 54v-5a30 30 0 0130-30h75a28 28 0 0128 28v7h77a30 30 0 0130 30v84a30 30 0 01-30 30H33a30 30 0 01-30-30V54z"/>
  </symbol>
</defs>
</svg>
"""

HTML_HEAD = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>artifacts</title>
</head>
<body>
{SVG_DEFS}
<table>
<tbody>
"""

HTML_FOOT = """
</tbody>
</table>
</body>
</html>
"""

def generate_index_html(directory):
    rows = []
    try:
        for entry in os.scandir(directory):
            if entry.name.startswith("."):
                continue
            rows.append(
                f'<tr><td><a href="{entry.name}">{entry.name}</a></td></tr>'
            )
    except PermissionError:
        return

    with open(os.path.join(directory, "index.html"), "w") as f:
        f.write(HTML_HEAD + "\n".join(rows) + HTML_FOOT)

def generate_indexes_recursive(root):
    for d, _, _ in os.walk(root):
        generate_index_html(d)

def run_command(cmd, cwd=None):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

def find_package_dir():
    base = os.path.join(os.getcwd(), "output", "packages")
    if not os.path.exists(base):
        raise RuntimeError(f"Package directory not found: {base}")
    return base

def mmddyy():
    return datetime.datetime.utcnow().strftime("%m%d%y")

def s3_object_exists(s3, bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise

def create_deb_repo(package_dir, origin):
    print("Creating APT repository...")

    dists = os.path.join(package_dir, "dists", "stable", "main", "binary-amd64")
    pool = os.path.join(package_dir, "pool", "main")

    os.makedirs(dists, exist_ok=True)
    os.makedirs(pool, exist_ok=True)

    for f in os.listdir(package_dir):
        if f.endswith(".deb"):
            shutil.move(os.path.join(package_dir, f), os.path.join(pool, f))

    run_command(
        "dpkg-scanpackages -m pool/main /dev/null > dists/stable/main/binary-amd64/Packages",
        cwd=package_dir,
    )
    run_command("gzip -9c Packages > Packages.gz", cwd=dists)

    release = os.path.join(package_dir, "dists", "stable", "Release")
    with open(release, "w") as f:
        f.write(
            f"""Origin: {origin}
Label: {origin}
Suite: stable
Codename: stable
Architectures: amd64
Components: main
Date: {datetime.datetime.utcnow():%a, %d %b %Y %H:%M:%S UTC}
"""
        )

    generate_indexes_recursive(package_dir)

def create_rpm_repo(package_dir):
    print("Creating RPM repository...")

    arch_dir = os.path.join(package_dir, "x86_64")
    os.makedirs(arch_dir, exist_ok=True)

    for f in os.listdir(package_dir):
        if f.endswith(".rpm"):
            shutil.move(os.path.join(package_dir, f), os.path.join(arch_dir, f))

    run_command("createrepo_c .", cwd=arch_dir)
    generate_indexes_recursive(package_dir)


def upload_to_s3(source_dir, bucket, prefix, dedupe=False):
    s3 = boto3.client("s3")
    print(f"Uploading to s3://{bucket}/{prefix}/")
    print(f"Deduplication: {'ON' if dedupe else 'OFF'}")

    skipped = 0
    uploaded = 0

    for root, _, files in os.walk(source_dir):
        for fname in files:
            local = os.path.join(root, fname)
            rel = os.path.relpath(local, source_dir)
            key = os.path.join(prefix, rel).replace("\\", "/")

            if dedupe and (fname.endswith(".deb") or fname.endswith(".rpm")):
                if s3_object_exists(s3, bucket, key):
                    print(f"Skipping existing package: {fname}")
                    skipped += 1
                    continue

            extra = {"ContentType": "text/html"} if fname.endswith(".html") else None

            print(f"Uploading: {key}")
            s3.upload_file(local, bucket, key, ExtraArgs=extra)
            uploaded += 1

    print(f"Uploaded: {uploaded}, Skipped: {skipped}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkg-type", required=True, choices=["deb", "rpm"])
    parser.add_argument("--s3-bucket", required=True)
    parser.add_argument("--amdgpu-family", required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument(
        "--job",
        default="ci",
        choices=["ci", "nightly"],
        help="Enable nightly shared repo",
    )

    args = parser.parse_args()
    package_dir = find_package_dir()

    if args.job == "nightly":
        prefix = f"{args.pkg_type}/{mmddyy()}"
        dedupe = True
    else:
        prefix = f"{args.amdgpu_family}_{args.artifact_id}/{args.pkg_type}"
        dedupe = False

    if args.pkg_type == "deb":
        create_deb_repo(package_dir, args.s3_bucket)
    else:
        create_rpm_repo(package_dir)

    upload_to_s3(package_dir, args.s3_bucket, prefix, dedupe=dedupe)


if __name__ == "__main__":
    main()
