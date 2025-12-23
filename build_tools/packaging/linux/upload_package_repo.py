#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Packaging + repository upload tool.

Usage:
python ./build_tools/packaging/linux/upload_package_repo.py \
             --pkg-type deb \
             --s3-bucket therock-deb-rpm-test \
             --amdgpu-family gfx94X-dcgpu \
             --artifact-id 16418185899 \
             --job nightly

Dev upload location:
  s3bucket/deb/<YYYYMMDD>-<artifact_id>
  s3bucket/rpm/<YYYYMMDD>-<artifact_id>

Nightly upload location:
  s3bucket/deb/<YYYYMMDD>-<artifact_id>
  s3bucket/rpm/<YYYYMMDD>-<artifact_id>
"""

import os
import argparse
import subprocess
import boto3
import shutil
import datetime
from pathlib import Path

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
            rows.append(f'<tr><td><a href="{entry.name}">{entry.name}</a></td></tr>')
    except PermissionError:
        return

    with open(os.path.join(directory, "index.html"), "w") as f:
        f.write(HTML_HEAD + "\n".join(rows) + HTML_FOOT)


def generate_indexes_recursive(root):
    for d, _, _ in os.walk(root):
        generate_index_html(d)


def regenerate_repo_metadata_from_s3(s3, bucket, prefix, pkg_type, local_package_dir):
    """Regenerate repository metadata efficiently using merge approach.

    This uses mergerepo_c (RPM) or merges Packages files (DEB) to efficiently
    update metadata without re-downloading all packages from S3.

    Args:
        s3: boto3 S3 client
        bucket: S3 bucket name
        prefix: S3 prefix (e.g., 'rpm/20251222-12345')
        pkg_type: Package type ('rpm' or 'deb')
        local_package_dir: Local directory with new packages
    """
    import tempfile

    print(f"Updating {pkg_type.upper()} repository metadata (merge mode)...")

    # Create temporary directory for metadata operations
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        if pkg_type == "rpm":
            # Efficient approach: Download existing repodata and merge with new packages
            old_repo_dir = temp_path / "old_repo"
            new_repo_dir = temp_path / "new_repo"
            merged_repo_dir = temp_path / "merged_repo"

            old_repo_dir.mkdir(parents=True, exist_ok=True)
            new_repo_dir.mkdir(parents=True, exist_ok=True)
            merged_repo_dir.mkdir(parents=True, exist_ok=True)

            # Step 1: Download existing repodata from S3 (small files)
            old_repodata_dir = old_repo_dir / "repodata"
            old_repodata_dir.mkdir(parents=True, exist_ok=True)

            print("Downloading existing repository metadata from S3...")
            repodata_files = []
            try:
                paginator = s3.get_paginator("list_objects_v2")
                for page in paginator.paginate(
                    Bucket=bucket, Prefix=f"{prefix}/x86_64/repodata/"
                ):
                    if "Contents" not in page:
                        continue
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        filename = Path(key).name
                        local_file = old_repodata_dir / filename
                        s3.download_file(bucket, key, str(local_file))
                        repodata_files.append(filename)
                        print(f"  Downloaded: {filename}")
            except Exception as e:
                print(f"Note: No existing repodata found (new repo?): {e}")

            # Step 2: Generate repodata for NEW packages only (from local dir)
            local_arch_dir = Path(local_package_dir) / "x86_64"
            if local_arch_dir.exists() and list(local_arch_dir.glob("*.rpm")):
                print("Generating metadata for new packages...")
                # Copy new RPMs to temp dir
                new_arch_dir = new_repo_dir / "x86_64"
                new_arch_dir.mkdir(parents=True, exist_ok=True)
                for rpm_file in local_arch_dir.glob("*.rpm"):
                    shutil.copy2(rpm_file, new_arch_dir / rpm_file.name)

                # Generate repodata for new packages
                run_command("createrepo_c .", cwd=str(new_arch_dir))
                print("✅ Generated metadata for new packages")
            else:
                print("No new RPM packages to process")
                return

            # Step 3: Merge repositories using mergerepo_c (no need to download all RPMs!)
            merged_arch_dir = merged_repo_dir / "x86_64"
            merged_arch_dir.mkdir(parents=True, exist_ok=True)

            if repodata_files:  # If we have existing metadata
                print("Merging old and new repository metadata...")
                # mergerepo_c merges repodata without needing actual RPM files!
                run_command(
                    f'mergerepo_c --repo "{old_repo_dir}" --repo "{new_repo_dir / "x86_64"}" --outputdir "{merged_arch_dir}"',
                    cwd=str(temp_path),
                )
                print("✅ Merged repository metadata")
            else:  # First upload, no existing metadata
                print("First upload - using new repository metadata")
                shutil.copytree(
                    new_repo_dir / "x86_64" / "repodata", merged_arch_dir / "repodata"
                )

            # Step 4: Upload merged repodata to S3
            merged_repodata = merged_arch_dir / "repodata"
            if merged_repodata.exists():
                print("Uploading merged repository metadata to S3...")
                for metadata_file in merged_repodata.iterdir():
                    if metadata_file.is_file():
                        s3_key = f"{prefix}/x86_64/repodata/{metadata_file.name}"
                        s3.upload_file(str(metadata_file), bucket, s3_key)
                        print(f"  Uploaded: {metadata_file.name}")
                print("✅ RPM repository metadata updated (merge complete)")

        elif pkg_type == "deb":
            # Efficient approach: Merge existing Packages file with new packages
            dists_dir = temp_path / "dists" / "stable" / "main" / "binary-amd64"
            dists_dir.mkdir(parents=True, exist_ok=True)

            pool_dir = temp_path / "pool" / "main"
            pool_dir.mkdir(parents=True, exist_ok=True)

            # Step 1: Download existing Packages file from S3 (small file)
            existing_packages = dists_dir / "Packages.old"
            try:
                print("Downloading existing Packages file from S3...")
                s3.download_file(
                    bucket,
                    f"{prefix}/dists/stable/main/binary-amd64/Packages",
                    str(existing_packages),
                )
                print("✅ Downloaded existing Packages file")
            except Exception as e:
                print(f"Note: No existing Packages file found (new repo?): {e}")
                existing_packages = None

            # Step 2: Generate Packages entries for NEW packages only (from local dir)
            local_pool_dir = Path(local_package_dir) / "pool" / "main"
            if local_pool_dir.exists() and list(local_pool_dir.glob("*.deb")):
                print("Generating Packages entries for new packages...")
                # Copy new DEBs to temp dir
                for deb_file in local_pool_dir.glob("*.deb"):
                    shutil.copy2(deb_file, pool_dir / deb_file.name)

                # Generate Packages entries for new packages
                new_packages = dists_dir / "Packages.new"
                run_command(
                    f'dpkg-scanpackages -m pool/main /dev/null > "{new_packages}"',
                    cwd=str(temp_path),
                )
                print("✅ Generated Packages entries for new packages")
            else:
                print("No new DEB packages to process")
                return

            # Step 3: Merge old and new Packages files
            merged_packages = dists_dir / "Packages"

            if existing_packages and existing_packages.exists():
                print("Merging old and new Packages files...")
                # Merge Packages files (no need to download all DEBs!)
                with open(merged_packages, "w") as outfile:
                    # Add existing packages
                    with open(existing_packages, "r") as infile:
                        outfile.write(infile.read())
                    # Add new packages
                    with open(new_packages, "r") as infile:
                        outfile.write(infile.read())
                print("✅ Merged Packages files")
            else:  # First upload, no existing Packages file
                print("First upload - using new Packages file")
                shutil.copy2(new_packages, merged_packages)

            # Compress Packages file
            run_command("gzip -9c Packages > Packages.gz", cwd=str(dists_dir))

            # Step 4: Upload merged Packages files to S3
            packages_file = dists_dir / "Packages"
            packages_gz = dists_dir / "Packages.gz"

            if packages_file.exists():
                s3_key = f"{prefix}/dists/stable/main/binary-amd64/Packages"
                s3.upload_file(str(packages_file), bucket, s3_key)
                print(f"  Uploaded: Packages")

            if packages_gz.exists():
                s3_key = f"{prefix}/dists/stable/main/binary-amd64/Packages.gz"
                s3.upload_file(str(packages_gz), bucket, s3_key)
                print(f"  Uploaded: Packages.gz")

            print("✅ DEB repository metadata updated (merge complete)")


def generate_top_index_from_s3(s3, bucket, prefix):
    """Generate index.html for top-level directory using S3 Delimiter.

    This is much more efficient than listing all objects recursively,
    as it only retrieves immediate subdirectories and files.

    Args:
        s3: boto3 S3 client
        bucket: S3 bucket name
        prefix: S3 prefix (e.g., 'deb' or 'rpm')
    """
    print(f"Generating top index from S3: s3://{bucket}/{prefix}/")

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/", Delimiter="/")

    rows = []

    for page in pages:
        # Add subdirectories (CommonPrefixes returned by Delimiter)
        for cp in page.get("CommonPrefixes", []):
            folder = cp["Prefix"][len(prefix) + 1 :].rstrip("/")
            rows.append(f'<tr><td><a href="{folder}/">{folder}/</a></td></tr>')

        # Add files at this level only (no nested files)
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            name = key[len(prefix) + 1 :]
            if "/" not in name:  # Only files at this level
                rows.append(f'<tr><td><a href="{name}">{name}</a></td></tr>')

    index_content = HTML_HEAD + "\n".join(rows) + HTML_FOOT
    index_key = f"{prefix}/index.html"

    print(f"Uploading top index: {index_key}")
    s3.put_object(
        Bucket=bucket,
        Key=index_key,
        Body=index_content.encode("utf-8"),
        ContentType="text/html",
    )
    print(f"✓ Successfully uploaded top-level index")


def generate_index_from_s3(s3, bucket, prefix, max_depth=None):
    """Generate index.html files based on what's actually in S3.

    This ensures index files accurately reflect the S3 repository state,
    including files from previous uploads that may have been deduplicated.

    Args:
        s3: boto3 S3 client
        bucket: S3 bucket name
        prefix: S3 prefix (e.g., 'deb/20251222')
        max_depth: Maximum directory depth to generate indexes for.
                   None = unlimited (recursive), 0 = only root level, 1 = root + immediate children
    """
    depth_msg = (
        f" (max depth: {max_depth})" if max_depth is not None else " (recursive)"
    )
    print(f"Generating indexes from S3: s3://{bucket}/{prefix}/{depth_msg}")

    # Get all objects under the prefix
    paginator = s3.get_paginator("list_objects_v2")
    all_objects = []

    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" not in page:
                continue
            all_objects.extend(page["Contents"])
    except Exception as e:
        print(f"Error listing S3 objects: {e}")
        return

    if not all_objects:
        print(f"No objects found in s3://{bucket}/{prefix}/")
        return

    # Group objects by directory
    directories = {}
    for obj in all_objects:
        key = obj["Key"]

        # Skip existing index.html files
        if key.endswith("index.html"):
            continue

        # Get the directory path relative to prefix
        if key.startswith(prefix):
            rel_path = key[len(prefix) :].lstrip("/")
        else:
            rel_path = key

        # Determine directory and filename
        if "/" in rel_path:
            dir_path = "/".join(rel_path.split("/")[:-1])
            filename = rel_path.split("/")[-1]
        else:
            dir_path = ""
            filename = rel_path

        if dir_path not in directories:
            directories[dir_path] = []
        directories[dir_path].append(filename)

        # Track all parent directories (even if they have no files, only subdirs)
        parts = dir_path.split("/") if dir_path else []
        for i in range(len(parts)):
            parent = "/".join(parts[:i])  # Empty string for root, or partial path
            if parent not in directories:
                directories[parent] = []  # Parent may have no files, only subdirs

    # Ensure root directory exists (in case all objects are in subdirectories)
    if "" not in directories:
        directories[""] = []

    # Generate index.html for each directory
    uploaded_indexes = 0
    for dir_path, files in sorted(directories.items()):
        # Check depth limit
        if max_depth is not None:
            # Calculate depth: empty string = 0, "a" = 0, "a/b" = 1, "a/b/c" = 2
            depth = dir_path.count("/") if dir_path else 0
            if depth > max_depth:
                continue  # Skip directories beyond max_depth

        # Create HTML rows
        rows = []

        # Add subdirectories first
        subdirs = set()
        for other_dir in directories.keys():
            if other_dir.startswith(dir_path + "/") and other_dir != dir_path:
                # Get immediate subdirectory
                remainder = other_dir[len(dir_path) :].lstrip("/")
                if "/" in remainder:
                    subdir = remainder.split("/")[0]
                else:
                    subdir = remainder
                if subdir:
                    subdirs.add(subdir)

        for subdir in sorted(subdirs):
            rows.append(f'<tr><td><a href="{subdir}/">{subdir}/</a></td></tr>')

        # Add files
        for filename in sorted(files):
            rows.append(f'<tr><td><a href="{filename}">{filename}</a></td></tr>')

        # Generate index.html content
        index_content = HTML_HEAD + "\n".join(rows) + HTML_FOOT

        # Determine the S3 key for this index.html
        if dir_path:
            index_key = f"{prefix}/{dir_path}/index.html"
        else:
            index_key = f"{prefix}/index.html"

        # Upload index.html to S3
        try:
            print(f"Uploading index: {index_key}")
            s3.put_object(
                Bucket=bucket,
                Key=index_key,
                Body=index_content.encode("utf-8"),
                ContentType="text/html",
            )
            uploaded_indexes += 1
        except Exception as e:
            print(f"Error uploading index {index_key}: {e}")

    print(f"Generated and uploaded {uploaded_indexes} index files from S3 state")


def run_command(cmd, cwd=None):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def find_package_dir():
    base = os.path.join(os.getcwd(), "output", "packages")
    if not os.path.exists(base):
        raise RuntimeError(f"Package directory not found: {base}")
    return base


def yyyymmdd():
    return datetime.datetime.utcnow().strftime("%Y%m%d")


def s3_object_exists(s3, bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def create_deb_repo(package_dir, job_type):
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
            f"""Origin: AMD ROCm
Label: ROCm {job_type} Packages
Suite: stable
Codename: stable
Architectures: amd64
Components: main
Date: {datetime.datetime.utcnow():%a, %d %b %Y %H:%M:%S UTC}
"""
        )

    # Index generation now happens from S3 state after upload


def create_rpm_repo(package_dir):
    """Create RPM repository structure.

    Note: Repository metadata (repodata) will be regenerated from S3 after upload
    to ensure it reflects all packages, including deduplicated ones.
    """
    print("Creating RPM repository...")

    arch_dir = os.path.join(package_dir, "x86_64")
    os.makedirs(arch_dir, exist_ok=True)

    for f in os.listdir(package_dir):
        if f.endswith(".rpm"):
            shutil.move(os.path.join(package_dir, f), os.path.join(arch_dir, f))

    # Generate initial repodata from local packages
    # This will be regenerated from S3 state after upload
    run_command("createrepo_c .", cwd=arch_dir)

    # Index generation now happens from S3 state after upload


def upload_to_s3(source_dir, bucket, prefix, dedupe=False):
    s3 = boto3.client("s3")
    print(f"Uploading to s3://{bucket}/{prefix}/")
    print(f"Deduplication: {'ON' if dedupe else 'OFF'}")

    skipped = 0
    uploaded = 0

    for root, _, files in os.walk(source_dir):
        for fname in files:
            # Skip index.html files - we'll generate them from S3 state
            if fname == "index.html":
                continue

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

    return s3  # Return S3 client for metadata regeneration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkg-type", required=True, choices=["deb", "rpm"])
    parser.add_argument("--s3-bucket", required=True)
    parser.add_argument("--amdgpu-family", required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument(
        "--job",
        default="dev",
        choices=["dev", "nightly"],
        help="Enable dev or nightly shared repo",
    )

    args = parser.parse_args()
    package_dir = find_package_dir()

    # TODO : Add the cases for release/prerelease
    if args.job in ["nightly", "dev"]:
        prefix = f"{args.pkg_type}/{yyyymmdd()}-{args.artifact_id}"
        dedupe = True

    if args.pkg_type == "deb":
        create_deb_repo(package_dir, args.job)
    else:
        create_rpm_repo(package_dir)

    # Upload packages and metadata to S3
    s3_client = upload_to_s3(package_dir, args.s3_bucket, prefix, dedupe=dedupe)

    # Efficiently update repository metadata by merging with existing metadata
    # (avoids re-downloading all packages from S3)
    regenerate_repo_metadata_from_s3(
        s3_client, args.s3_bucket, prefix, args.pkg_type, package_dir
    )

    # Generate index.html files from S3 state (recursive for specific upload)
    generate_index_from_s3(s3_client, args.s3_bucket, prefix)

    # Generate a top-level index for the pkg type (e.g., 'deb' or 'rpm')
    # Uses S3 Delimiter for efficiency (only lists folders, not all nested files)
    top_prefix = prefix.split("/")[0]
    generate_top_index_from_s3(s3_client, args.s3_bucket, top_prefix)


if __name__ == "__main__":
    main()
