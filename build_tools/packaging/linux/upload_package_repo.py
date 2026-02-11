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


def regenerate_rpm_metadata_from_s3(s3, bucket, prefix, uploaded_packages):
    """Regenerate RPM repository metadata using merge approach.

    Downloads existing repodata from S3, generates metadata for new packages,
    merges them using mergerepo_c, and uploads the result back to S3.

    Args:
        s3: boto3 S3 client
        bucket: S3 bucket name
        prefix: S3 prefix (e.g., 'rpm/20251222-12345')
        uploaded_packages: List of actually uploaded .rpm file paths
    """
    import tempfile

    print(f"Updating RPM repository metadata (merge mode)...")

    # Create temporary directory for metadata operations
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

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

        print(
            f"Downloading existing repository metadata from S3: s3://{bucket}/{prefix}/x86_64/repodata/"
        )
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
            if repodata_files:
                print(
                    f"✅ Found {len(repodata_files)} existing metadata files to merge"
                )
            else:
                print("No existing metadata files found")
        except Exception as e:
            print(f"⚠️  No existing repodata found (new repo?): {e}")

        # Step 2: Generate repodata for NEW packages only (actually uploaded ones)
        rpm_packages = [p for p in uploaded_packages if p.endswith(".rpm")]
        if rpm_packages:
            print(
                f"Generating metadata for {len(rpm_packages)} uploaded RPM packages..."
            )
            # Copy uploaded RPMs to temp dir
            new_arch_dir = new_repo_dir / "x86_64"
            new_arch_dir.mkdir(parents=True, exist_ok=True)
            for rpm_file in rpm_packages:
                shutil.copy2(rpm_file, new_arch_dir / Path(rpm_file).name)

            # Generate repodata for new packages with clean paths (no baseurl)
            run_command(
                "createrepo_c --no-database --simple-md-filenames .",
                cwd=str(new_arch_dir),
            )
            print("✅ Generated metadata for uploaded packages")
        else:
            print("No new RPM packages uploaded (all deduplicated)")
            # Still need to ensure old metadata is preserved!
            if repodata_files:
                print("Preserving existing repodata...")
                # Just re-upload the existing repodata we downloaded
                for metadata_file in old_repodata_dir.iterdir():
                    if metadata_file.is_file():
                        s3_key = f"{prefix}/x86_64/repodata/{metadata_file.name}"
                        s3.upload_file(str(metadata_file), bucket, s3_key)
                        print(f"  Uploaded: {metadata_file.name}")
                print("✅ RPM repository metadata preserved")
            return

        # Step 3: Merge repositories using mergerepo_c (no need to download all RPMs!)
        merged_arch_dir = merged_repo_dir / "x86_64"
        merged_arch_dir.mkdir(parents=True, exist_ok=True)

        if repodata_files:  # If we have existing metadata
            print("Merging old and new repository metadata...")
            # mergerepo_c merges repodata without needing actual RPM files!
            # Use --no-database, --simple-md-filenames, and --omit-baseurl to ensure clean paths
            run_command(
                f"mergerepo_c --no-database --simple-md-filenames --omit-baseurl "
                f'--repo "{old_repo_dir}" --repo "{new_repo_dir / "x86_64"}" '
                f'--outputdir "{merged_arch_dir}"',
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
            uploaded_metadata = []
            for metadata_file in merged_repodata.iterdir():
                if metadata_file.is_file():
                    s3_key = f"{prefix}/x86_64/repodata/{metadata_file.name}"
                    s3.upload_file(str(metadata_file), bucket, s3_key)
                    uploaded_metadata.append(metadata_file.name)
                    print(f"  Uploaded: {metadata_file.name}")
            print(f"✅ RPM repository metadata updated: {len(uploaded_metadata)} files")


def generate_release_file_with_checksums(release_file, job_type, dists_dir):
    """Generate a Debian Release file with MD5Sum, SHA1, and SHA256 checksums.

    Args:
        release_file: Path to the Release file to create
        job_type: Job type for metadata (nightly/dev/release)
        dists_dir: Directory containing Packages files (main/binary-amd64/)
    """
    import hashlib
    import datetime

    # Files to hash (relative paths from dists/stable/)
    files_to_hash = [
        (dists_dir / "Packages", "main/binary-amd64/Packages"),
        (dists_dir / "Packages.gz", "main/binary-amd64/Packages.gz"),
    ]

    # Calculate all hashes
    md5_entries = []
    sha1_entries = []
    sha256_entries = []

    for file_path, rel_path in files_to_hash:
        if not file_path.exists():
            continue

        # Get file size
        file_size = file_path.stat().st_size

        # Calculate hashes
        md5_hash = hashlib.md5()
        sha1_hash = hashlib.sha1()
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            while True:
                data = f.read(65536)  # Read in 64KB chunks
                if not data:
                    break
                md5_hash.update(data)
                sha1_hash.update(data)
                sha256_hash.update(data)

        # Store entries (space-aligned format)
        md5_entries.append(f" {md5_hash.hexdigest()} {file_size:16d} {rel_path}")
        sha1_entries.append(f" {sha1_hash.hexdigest()} {file_size:16d} {rel_path}")
        sha256_entries.append(f" {sha256_hash.hexdigest()} {file_size:16d} {rel_path}")

    # Write Release file
    with open(release_file, "w") as f:
        # Header fields
        f.write(
            f"""Origin: AMD ROCm
Label: ROCm {job_type} Packages
Suite: stable
Codename: stable
Architectures: amd64
Components: main
Description: ROCm APT Repository
Date: {datetime.datetime.utcnow():%a, %d %b %Y %H:%M:%S UTC}
"""
        )

        # MD5Sum section
        if md5_entries:
            f.write("MD5Sum:\n")
            f.write("\n".join(md5_entries))
            f.write("\n")

        # SHA1 section
        if sha1_entries:
            f.write("SHA1:\n")
            f.write("\n".join(sha1_entries))
            f.write("\n")

        # SHA256 section
        if sha256_entries:
            f.write("SHA256:\n")
            f.write("\n".join(sha256_entries))
            f.write("\n")

    print(f"✅ Release file generated with checksums: MD5, SHA1, SHA256")


def upload_deb_metadata_to_s3(s3, bucket, prefix, dists_dir, release_file):
    """Helper function to upload Debian metadata files to S3.

    Args:
        s3: boto3 S3 client
        bucket: S3 bucket name
        prefix: S3 prefix
        dists_dir: Directory containing Packages files
        release_file: Path to Release file
    """
    packages_file = dists_dir / "Packages"
    packages_gz = dists_dir / "Packages.gz"

    uploaded_count = 0
    if packages_file.exists():
        s3_key = f"{prefix}/dists/stable/main/binary-amd64/Packages"
        s3.upload_file(str(packages_file), bucket, s3_key)
        print(f"  Uploaded: Packages")
        uploaded_count += 1

    if packages_gz.exists():
        s3_key = f"{prefix}/dists/stable/main/binary-amd64/Packages.gz"
        s3.upload_file(str(packages_gz), bucket, s3_key)
        print(f"  Uploaded: Packages.gz")
        uploaded_count += 1

    if release_file.exists():
        s3_key = f"{prefix}/dists/stable/Release"
        s3.upload_file(str(release_file), bucket, s3_key)
        print(f"  Uploaded: Release")
        uploaded_count += 1

    print(f"✅ DEB repository metadata updated: {uploaded_count} files")


def regenerate_deb_metadata_from_s3(
    s3, bucket, prefix, uploaded_packages, job_type="nightly"
):
    """Regenerate Debian repository metadata efficiently with proper checksums.

    Uses dpkg-scanpackages for efficiency (no package downloads), but generates
    proper Release file with MD5Sum, SHA1, and SHA256 checksums.

    Args:
        s3: boto3 S3 client
        bucket: S3 bucket name
        prefix: S3 prefix (e.g., 'deb/20251222-12345')
        uploaded_packages: List of actually uploaded .deb file paths
        job_type: Job type for Release file metadata (default: 'nightly')
    """
    import tempfile

    print(f"Updating DEB repository metadata (merge mode with checksums)...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Setup directories
        dists_dir = temp_path / "dists" / "stable" / "main" / "binary-amd64"
        dists_dir.mkdir(parents=True, exist_ok=True)

        pool_dir = temp_path / "pool" / "main"
        pool_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Download existing Packages file from S3 (SMALL FILE - efficient!)
        existing_packages = dists_dir / "Packages.old"
        packages_s3_key = f"{prefix}/dists/stable/main/binary-amd64/Packages"
        try:
            print(
                f"Downloading existing Packages file from S3: s3://{bucket}/{packages_s3_key}"
            )
            s3.download_file(bucket, packages_s3_key, str(existing_packages))
            with open(existing_packages, "r") as f:
                content = f.read()
                pkg_count = content.count("\nPackage: ")
            print(f"✅ Downloaded existing Packages file ({pkg_count} packages)")
        except Exception as e:
            print(f"⚠️  No existing Packages file found (new repo?): {e}")
            existing_packages = None

        # Step 2: Generate Packages entries for NEW packages only
        deb_packages = [p for p in uploaded_packages if p.endswith(".deb")]
        if deb_packages:
            print(
                f"Generating Packages entries for {len(deb_packages)} uploaded DEB packages..."
            )
            # Copy uploaded DEBs to temp dir
            for deb_file in deb_packages:
                shutil.copy2(deb_file, pool_dir / Path(deb_file).name)

            # Generate Packages entries for uploaded packages
            new_packages = dists_dir / "Packages.new"
            run_command(
                f'dpkg-scanpackages -m pool/main /dev/null > "{new_packages}"',
                cwd=str(temp_path),
            )
            print("✅ Generated Packages entries for uploaded packages")
        else:
            print("No new DEB packages uploaded (all deduplicated)")
            if existing_packages and existing_packages.exists():
                print("Preserving existing metadata...")
                shutil.copy2(existing_packages, dists_dir / "Packages")
                run_command("gzip -9c Packages > Packages.gz", cwd=str(dists_dir))

                # Generate Release file with checksums
                release_dir = temp_path / "dists" / "stable"
                release_dir.mkdir(parents=True, exist_ok=True)
                release_file = release_dir / "Release"

                generate_release_file_with_checksums(release_file, job_type, dists_dir)

                # Upload preserved files
                upload_deb_metadata_to_s3(s3, bucket, prefix, dists_dir, release_file)
            return

        # Step 3: Merge old and new Packages files
        merged_packages = dists_dir / "Packages"

        if existing_packages and existing_packages.exists():
            print("Merging old and new Packages files...")

            def parse_packages_file(filepath):
                """Parse Packages file into dict keyed by Filename"""
                packages = {}
                with open(filepath, "r") as f:
                    current_entry = []
                    current_filename = None

                    for line in f:
                        if line.strip() == "":
                            if current_entry and current_filename:
                                packages[current_filename] = (
                                    "\n".join(current_entry) + "\n"
                                )
                            current_entry = []
                            current_filename = None
                        else:
                            current_entry.append(line.rstrip())
                            if line.startswith("Filename:"):
                                current_filename = line.split(":", 1)[1].strip()

                    if current_entry and current_filename:
                        packages[current_filename] = "\n".join(current_entry) + "\n"

                return packages

            old_packages = parse_packages_file(existing_packages)
            new_packages_dict = parse_packages_file(new_packages)

            print(f"  Old metadata: {len(old_packages)} packages")
            print(f"  New metadata: {len(new_packages_dict)} packages")

            merged = old_packages.copy()
            merged.update(new_packages_dict)

            with open(merged_packages, "w") as outfile:
                for filename in sorted(merged.keys()):
                    outfile.write(merged[filename])
                    outfile.write("\n")

            print(f"✅ Merged Packages files: {len(merged)} total packages")
        else:
            print("First upload - using new Packages file")
            shutil.copy2(new_packages, merged_packages)

        # Compress Packages file
        run_command("gzip -9c Packages > Packages.gz", cwd=str(dists_dir))

        # Step 4: Generate Release file with checksums
        release_dir = temp_path / "dists" / "stable"
        release_dir.mkdir(parents=True, exist_ok=True)
        release_file = release_dir / "Release"

        generate_release_file_with_checksums(release_file, job_type, dists_dir)

        # Step 5: Upload merged files to S3
        upload_deb_metadata_to_s3(s3, bucket, prefix, dists_dir, release_file)


def regenerate_repo_metadata_from_s3(
    s3, bucket, prefix, pkg_type, uploaded_packages, job_type="nightly"
):
    """Regenerate repository metadata efficiently using merge approach.

    This uses mergerepo_c (RPM) or merges Packages files (DEB) to efficiently
    update metadata without re-downloading all packages from S3.

    Args:
        s3: boto3 S3 client
        bucket: S3 bucket name
        prefix: S3 prefix (e.g., 'rpm/20251222-12345')
        pkg_type: Package type ('rpm' or 'deb')
        uploaded_packages: List of actually uploaded package file paths (avoids duplicates from deduplication)
        job_type: Job type for Release file metadata (default: 'nightly')
    """
    if pkg_type == "rpm":
        regenerate_rpm_metadata_from_s3(s3, bucket, prefix, uploaded_packages)
    elif pkg_type == "deb":
        regenerate_deb_metadata_from_s3(s3, bucket, prefix, uploaded_packages, job_type)
    else:
        raise ValueError(f"Unsupported package type: {pkg_type}")


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
            rows.append(
                f'<tr><td><a href="{folder}/index.html">{folder}/</a></td></tr>'
            )

        # Add files at this level only (no nested files)
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/") or key.endswith("index.html"):
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
    # Sort by depth (deepest first), then alphabetically
    # This ensures leaf directories are created before parent directories
    uploaded_indexes = 0
    for dir_path, files in sorted(
        directories.items(), key=lambda x: (-x[0].count("/") if x[0] else 1, x[0])
    ):
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
            # Handle root directory (empty string) specially
            if dir_path == "":
                # Any other_dir is potentially a subdirectory of root
                if other_dir:  # Not empty
                    if "/" in other_dir:
                        # Multi-level path: get first component
                        subdir = other_dir.split("/")[0]
                    else:
                        # Single-level path: it's a direct child
                        subdir = other_dir
                    subdirs.add(subdir)
            else:
                # Non-root: check if other_dir is under this dir_path
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
            rows.append(
                f'<tr><td><a href="{subdir}/index.html">{subdir}/</a></td></tr>'
            )

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

    # Generate initial repodata from local packages with clean paths (no baseurl)
    # This will be regenerated from S3 state after upload
    run_command("createrepo_c --no-database --simple-md-filenames .", cwd=arch_dir)

    # Index generation now happens from S3 state after upload


def upload_to_s3(source_dir, bucket, prefix, dedupe=False):
    s3 = boto3.client("s3")
    print(f"Uploading to s3://{bucket}/{prefix}/")
    print(f"Deduplication: {'ON' if dedupe else 'OFF'}")

    skipped = 0
    uploaded = 0
    uploaded_packages = []  # Track actually uploaded package files

    for root, _, files in os.walk(source_dir):
        for fname in files:
            # Skip index.html files - we'll generate them from S3 state
            if fname == "index.html":
                continue

            # Skip build manifest files - these are for local tracking only
            if fname.lower().endswith(".txt"):
                print(f"Skipping build manifest file (local only): {fname}")
                continue

            local = os.path.join(root, fname)
            rel = os.path.relpath(local, source_dir)
            key = os.path.join(prefix, rel).replace("\\", "/")

            # Skip metadata files - they'll be regenerated/merged properly later
            # For DEB: skip Packages, Packages.gz, Release in dists/
            # For RPM: skip repodata/* files
            if "/repodata/" in key or key.endswith("/repodata"):
                print(f"Skipping metadata file (will regenerate): {fname}")
                continue
            if "/dists/" in key and (
                fname in ["Packages", "Packages.gz", "Release", "InRelease"]
            ):
                print(f"Skipping metadata file (will regenerate): {fname}")
                continue

            if dedupe and (fname.endswith(".deb") or fname.endswith(".rpm")):
                if s3_object_exists(s3, bucket, key):
                    print(f"Skipping existing package: {fname}")
                    skipped += 1
                    continue

            extra = {"ContentType": "text/html"} if fname.endswith(".html") else None

            print(f"Uploading: {key}")
            s3.upload_file(local, bucket, key, ExtraArgs=extra)
            uploaded += 1

            # Track uploaded packages for metadata generation
            if fname.endswith(".deb") or fname.endswith(".rpm"):
                uploaded_packages.append(local)

    print(f"Uploaded: {uploaded}, Skipped: {skipped}")
    if uploaded_packages:
        print(f"Uploaded packages: {[Path(p).name for p in uploaded_packages]}")

    return s3, uploaded_packages  # Return S3 client and list of uploaded packages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkg-type", required=True, choices=["deb", "rpm"])
    parser.add_argument("--s3-bucket", required=True)
    parser.add_argument("--amdgpu-family", required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument(
        "--job",
        default="dev",
        choices=["dev", "nightly", "prerelease"],
        help="Enable dev or nightly shared repo",
    )

    args = parser.parse_args()
    package_dir = find_package_dir()

    # TODO : Add the cases for release/prerelease
    if args.job in ["nightly", "dev"]:
        prefix = f"{args.pkg_type}/{yyyymmdd()}-{args.artifact_id}"
        dedupe = True
    elif args.job == "prerelease":
        prefix = f"v3/packages/{args.pkg_type}"
        dedupe = True

    if args.pkg_type == "deb":
        create_deb_repo(package_dir, args.job)
    else:
        create_rpm_repo(package_dir)

    # Upload packages and metadata to S3
    s3_client, uploaded_packages = upload_to_s3(
        package_dir, args.s3_bucket, prefix, dedupe=dedupe
    )

    # Efficiently update repository metadata by merging with existing metadata
    # (avoids re-downloading all packages from S3)
    # Only generates metadata for actually uploaded packages (avoids duplicates from deduplication)
    regenerate_repo_metadata_from_s3(
        s3_client, args.s3_bucket, prefix, args.pkg_type, uploaded_packages, args.job
    )

    # Generate index.html files from S3 state (recursive for specific upload)
    generate_index_from_s3(s3_client, args.s3_bucket, prefix)

    # Generate a top-level index for the pkg type (e.g., 'deb' or 'rpm')
    # Uses S3 Delimiter for efficiency (only lists folders, not all nested files)
    top_prefix = prefix.split("/")[0]
    generate_top_index_from_s3(s3_client, args.s3_bucket, top_prefix)


if __name__ == "__main__":
    main()
