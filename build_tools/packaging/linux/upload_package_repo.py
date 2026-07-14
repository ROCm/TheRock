#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Packaging + repository upload tool.

Usage (multi-arch CI — new-style, preferred):
  python ./build_tools/packaging/linux/upload_package_repo.py \
    --pkg-type deb \
    --run-id 16418185899

  Bucket + prefix are resolved automatically via WorkflowOutputRoot using
  the GITHUB_REPOSITORY and RELEASE_TYPE environment variables:
    - CI builds    → therock-ci-artifacts / <run_id>-linux/packages/deb
    - dev builds   → therock-dev-artifacts / <run_id>-linux/packages/deb
    - nightly      → therock-nightly-artifacts / <run_id>-linux/packages/deb
    - prerelease   → therock-prerelease-artifacts / <run_id>-linux/packages/deb

Usage (single-arch release — legacy, build_native_linux_packages.yml):
  python ./build_tools/packaging/linux/upload_package_repo.py \
    --pkg-type deb \
    --s3-bucket therock-nightly-packages \
    --amdgpu-family gfx94X-dcgpu \
    --artifact-id 16418185899 \
    --job nightly

  Legacy upload locations:
    dev/nightly  → s3bucket/<pkg_type>/<YYYYMMDD>-<artifact_id>
    prerelease   → s3bucket/v3/packages/<pkg_type>
"""

import argparse
import boto3
import datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_BUILD_TOOLS_DIR = _THIS_DIR.parent.parent

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_BUILD_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from _therock_utils.storage_location import StorageLocation
from _therock_utils.workflow_outputs import WorkflowOutputRoot


def _rpm_arch_dir(package_dir: Path | None) -> Path | None:
    """Return package_dir/x86_64 when it exists."""
    if package_dir is None:
        return None
    arch_dir = package_dir / "x86_64"
    return arch_dir if arch_dir.is_dir() else None


def _local_rpm_names(arch_dir: Path) -> set[str]:
    return {path.name for path in arch_dir.glob("*.rpm")}


def _list_s3_rpm_keys(s3, bucket: str, prefix: str) -> list[str]:
    """List all RPM object keys under prefix/x86_64/."""
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/x86_64/"):
        if "Contents" not in page:
            continue
        for obj in page["Contents"]:
            key = obj["Key"]
            if key.endswith(".rpm"):
                keys.append(key)
    return keys


def _count_primary_packages(primary_xml_gz: Path) -> int:
    import gzip
    import xml.etree.ElementTree as ET

    root = ET.fromstring(gzip.open(primary_xml_gz, "rb").read())
    if root.tag.startswith("{"):
        namespace = root.tag.split("}")[0][1:]
        return len(root.findall(f"{{{namespace}}}package"))
    return len(root.findall("package"))


def _validate_rpm_repodata(arch_dir: Path) -> None:
    """Fail fast when repodata does not index every RPM in the arch dir."""
    rpm_count = len(list(arch_dir.glob("*.rpm")))
    primary_xml_gz = arch_dir / "repodata" / "primary.xml.gz"
    if not primary_xml_gz.exists():
        raise RuntimeError(f"Missing repodata primary.xml.gz under {arch_dir}")

    meta_count = _count_primary_packages(primary_xml_gz)
    if meta_count != rpm_count:
        raise RuntimeError(
            "RPM repodata is incomplete: "
            f"{meta_count} packages indexed, {rpm_count} .rpm files present"
        )


def _prepare_rpm_arch_dir_for_repodata(
    s3,
    bucket: str,
    prefix: str,
    package_dir: Path | None,
    work_dir: Path,
) -> tuple[Path, int, int]:
    """Materialize a complete x86_64 RPM directory for createrepo_c.

    Uses the local build tree first (includes dedupe-skipped packages), then
    downloads any RPMs that exist on S3 but not locally (prior uploads).

    Returns:
        Tuple of (arch_dir, local_rpm_count, downloaded_rpm_count)
    """
    arch_dir = work_dir / "x86_64"
    arch_dir.mkdir(parents=True, exist_ok=True)

    local_arch_dir = _rpm_arch_dir(package_dir)
    local_count = 0
    if local_arch_dir is not None:
        for rpm_file in local_arch_dir.glob("*.rpm"):
            shutil.copy2(rpm_file, arch_dir / rpm_file.name)
            local_count += 1

    local_names = _local_rpm_names(arch_dir)
    downloaded_count = 0
    for key in _list_s3_rpm_keys(s3, bucket, prefix):
        filename = Path(key).name
        if filename in local_names:
            continue
        print(f"  Downloading existing S3 package for repodata: {filename}")
        s3.download_file(bucket, key, str(arch_dir / filename))
        downloaded_count += 1

    return arch_dir, local_count, downloaded_count


def regenerate_rpm_metadata_from_s3(
    s3, bucket, prefix, uploaded_packages, package_dir=None
):
    """Regenerate RPM repository metadata from the full package set on S3.

    Runs createrepo_c over every .rpm under prefix/x86_64/: the local build
    output (even when upload dedupe skipped uploading them) plus any RPMs that
    already exist on S3 from prior uploads.

    Args:
        s3: boto3 S3 client
        bucket: S3 bucket name
        prefix: S3 prefix (e.g., 'rpm/20251222-12345')
        uploaded_packages: List of .rpm file paths uploaded this run (unused;
            kept for API compatibility with regenerate_repo_metadata_from_s3)
        package_dir: Local package tree produced by build_package.py
    """
    import tempfile

    del uploaded_packages  # full regen uses local tree + S3 listing, not this subset

    print("Updating RPM repository metadata (full regen from local + S3)...")

    with tempfile.TemporaryDirectory() as temp_dir:
        arch_dir, local_count, downloaded_count = _prepare_rpm_arch_dir_for_repodata(
            s3,
            bucket,
            prefix,
            Path(package_dir) if package_dir is not None else None,
            Path(temp_dir),
        )
        rpm_count = len(list(arch_dir.glob("*.rpm")))
        if rpm_count == 0:
            print("No RPM packages found locally or on S3; skipping repodata regen")
            return

        print(
            f"Generating repodata for {rpm_count} RPM packages "
            f"({local_count} local, {downloaded_count} downloaded from S3)..."
        )
        run_command(
            "createrepo_c --no-database --simple-md-filenames .",
            cwd=str(arch_dir),
        )
        _validate_rpm_repodata(arch_dir)
        print("✅ Generated repodata for full RPM set")

        repodata_dir = arch_dir / "repodata"
        print("Uploading repository metadata to S3...")
        uploaded_metadata = []
        for metadata_file in repodata_dir.iterdir():
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
    s3,
    bucket,
    prefix,
    pkg_type,
    uploaded_packages,
    job_type="nightly",
    package_dir=None,
):
    """Regenerate repository metadata efficiently using merge approach.

    RPM repos are rebuilt from the full local + S3 package set. DEB repos
    continue to merge Packages files without re-downloading all packages.

    Args:
        s3: boto3 S3 client
        bucket: S3 bucket name
        prefix: S3 prefix (e.g., 'rpm/20251222-12345')
        pkg_type: Package type ('rpm' or 'deb')
        uploaded_packages: List of actually uploaded package file paths (avoids duplicates from deduplication)
        job_type: Job type for Release file metadata (default: 'nightly')
        package_dir: Local package tree produced by build_package.py
    """
    if pkg_type == "rpm":
        regenerate_rpm_metadata_from_s3(
            s3, bucket, prefix, uploaded_packages, package_dir=package_dir
        )
    elif pkg_type == "deb":
        regenerate_deb_metadata_from_s3(s3, bucket, prefix, uploaded_packages, job_type)
    else:
        raise ValueError(f"Unsupported package type: {pkg_type}")


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


def upload_to_s3(source_dir, bucket, prefix, dedupe=False):
    s3 = boto3.client("s3")
    print(f"Uploading to s3://{bucket}/{prefix}/")
    print(f"Deduplication: {'ON' if dedupe else 'OFF'}")

    skipped = 0
    uploaded = 0
    uploaded_packages = []  # Track actually uploaded package files

    for root, _, files in os.walk(source_dir):
        for fname in files:
            # Always skip local index.html files, those are generated server-side.
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


def _emit_github_output(key: str, value: str) -> None:
    """Write a key=value pair to $GITHUB_OUTPUT if running in GitHub Actions."""
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"{key}={value}\n")


def _package_install_url(bucket: str, prefix: str, pkg_type: str) -> str:
    """Compute the package manager install URL for a given repo location.

    For RPM repos, dnf/yum baseurl must point to the x86_64/ subdirectory
    (the directory containing repodata/). For DEB repos, apt points to the
    repo root (it resolves dists/ itself).
    """
    base = StorageLocation(bucket, prefix).https_url
    if pkg_type == "rpm":
        return f"{base}/x86_64"
    return base


def _resolve_upload_target(
    args: argparse.Namespace,
    pkg_type: str,
) -> tuple[str, str, str, bool, str]:
    """Resolve S3 bucket, prefix, install URL, dedupe flag, and job type.

    Returns:
        Tuple of (bucket, prefix, install_url, dedupe, job_type)
    """
    if args.run_id:
        # New-style: derive bucket + prefix from WorkflowOutputRoot.
        # This is the single source of truth for CI path layout.
        root = WorkflowOutputRoot.from_workflow_run(
            run_id=args.run_id, platform="linux"
        )
        loc = root.native_linux_packages(pkg_type)
        job_type = os.environ.get("RELEASE_TYPE", "ci")
        install_url = _package_install_url(loc.bucket, loc.relative_path, pkg_type)
        return loc.bucket, loc.relative_path, install_url, True, job_type

    if args.s3_prefix:
        # Legacy: explicit prefix provided by get_s3_config.py
        return (
            args.s3_bucket,
            args.s3_prefix,
            _package_install_url(args.s3_bucket, args.s3_prefix, pkg_type),
            True,
            args.job,
        )

    if args.job in ("nightly", "dev"):
        prefix = f"{pkg_type}/{yyyymmdd()}-{args.artifact_id}"
        return (
            args.s3_bucket,
            prefix,
            _package_install_url(args.s3_bucket, prefix, pkg_type),
            True,
            args.job,
        )

    if args.job == "prerelease":
        prefix = f"v3/packages/{pkg_type}"
        return (
            args.s3_bucket,
            prefix,
            _package_install_url(args.s3_bucket, prefix, pkg_type),
            True,
            args.job,
        )

    raise ValueError(f"Unknown job type: {args.job!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkg-type", required=True, choices=["deb", "rpm"])

    # New-style args: use WorkflowOutputRoot for bucket/prefix resolution.
    # When --run-id is provided, bucket and prefix are derived automatically
    # from CI context (GITHUB_REPOSITORY, RELEASE_TYPE, fork detection).
    parser.add_argument(
        "--run-id",
        help="GitHub Actions workflow run ID (enables WorkflowOutputRoot path resolution)",
    )

    # Legacy args: kept for backward compatibility with build_native_linux_packages.yml
    parser.add_argument("--s3-bucket")
    parser.add_argument("--amdgpu-family", required=False)  # unused, kept for compat
    parser.add_argument("--artifact-id")
    parser.add_argument(
        "--job",
        default="dev",
        choices=["dev", "nightly", "prerelease"],
        help="Job type (legacy, used when --run-id is not provided)",
    )
    parser.add_argument(
        "--s3-prefix",
        required=False,
        help="Override S3 prefix (legacy, used when --run-id is not provided)",
    )
    parser.add_argument(
        "--package-dir",
        required=True,
        help="Path to the directory containing built packages.",
    )

    args = parser.parse_args()
    package_dir = Path(args.package_dir).resolve()

    bucket, prefix, install_url, dedupe, job_type = _resolve_upload_target(
        args, args.pkg_type
    )

    if args.pkg_type == "deb":
        create_deb_repo(package_dir, job_type)
    else:
        create_rpm_repo(package_dir)

    # Upload packages and metadata to S3
    s3_client, uploaded_packages = upload_to_s3(
        package_dir, bucket, prefix, dedupe=dedupe
    )

    # Regenerate repository metadata from the full package set on S3/local tree
    regenerate_repo_metadata_from_s3(
        s3_client,
        bucket,
        prefix,
        args.pkg_type,
        uploaded_packages,
        job_type,
        package_dir=package_dir,
    )

    print(f"Package repository URL: {install_url}")
    _emit_github_output("package_repository_url", install_url)


if __name__ == "__main__":
    main()
