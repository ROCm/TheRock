#!/usr/bin/env python
"""Download prerelease packages from S3 bucket for promotion to release.

This script downloads release candidate packages from the therock-prerelease-python
S3 bucket. It discovers all architectures that contain packages matching a specific
version pattern and downloads them into a local directory structure.

In addition to downloading packages to promote, the script also allows to download
PyPI dependencies that are part of the same S3 bucket.

Selection of those packages can be done by version pattern and architecture.
Options to just list architectures found matching the version or list all packages
per architecture without downloading are also available. Packages that are not known
to be either a package to promote or a PyPI dependency are also listed.

Previously downloaded packages are skipped.

PREREQUISITES:
  - pip install -r ./build_tools/packaging/requirements.txt
  - AWS credentials configured
    - read and list bucket access for therock-prerelease-python bucket

TYPICAL USAGE (Command Line):
  # Download all 7.10.0rc2 packages for all architectures:
  python ./build_tools/packaging/download_prerelease_packages.py \
    --version=7.10.0rc2 \
    --output-dir=./downloads

  # Download only for a specific architecture:
  python ./build_tools/packaging/download_prerelease_packages.py \
    --version=7.10.0rc2 \
    --arch=gfx950-dcgpu \
    --output-dir=./downloads

  # Download for multiple specific architectures:
  python ./build_tools/packaging/download_prerelease_packages.py \
    --version=7.10.0rc2 \
    --arch=gfx1151,gfx950-dcgpu \
    --output-dir=./downloads

  # Download packages including tarballs:
  python ./build_tools/packaging/download_prerelease_packages.py \
    --version=7.10.0rc2 \
    --arch=gfx1151,gfx950-dcgpu \
    --output-dir=./downloads \
    --include-tarballs

  # List available architectures without downloading:
  python ./build_tools/packaging/download_prerelease_packages.py \
    --version=7.10.0rc2 \
    --list-archs

  # List all packages per architecture without downloading:
  python ./build_tools/packaging/download_prerelease_packages.py \
    --version=7.10.0rc2 \
    --list-packages-per-arch

  # List all packages and tarballs per architecture with sizes:
  python ./build_tools/packaging/download_prerelease_packages.py \
    --version=7.10.0rc2 \
    --list-packages-per-arch \
    --include-tarballs

  # Download packages to promote and all known PyPI dependencies:
  python ./build_tools/packaging/download_prerelease_packages.py \
    --version=7.10.0rc2 \
    --output-dir=./downloads \
    --include-dependencies

DIRECTORY STRUCTURE:
  Output directory structure will be:
    <output-dir>/
      <arch1>/
        package1.whl
        package2.whl
        ...
      <arch2>/
        package1.whl
        ...
      tarball/  (if --include-tarballs is specified)
        therock-dist-linux-<arch1>-<version>.tar.gz
        therock-dist-windows-<arch2>-<version>.tar.gz
        ...

PACKAGE CATEGORIES:
  Packages to promote (ROCm and PyTorch):
    - pytorch-triton-rocm
    - rocm (sdist)
    - rocm-sdk-core
    - rocm-sdk-devel
    - rocm-sdk-libraries-*
    - torch
    - torchaudio
    - torchvision

  Known dependencies:
    - filelock, fsspec, jinja2, markupsafe, mpmath, networkx
    - numpy, pillow, setuptools, sympy, typing-extensions

  Unknown packages:
    - everything else that is not a package to promote or a dependency package
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Union, Dict

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    print("[ERROR]: boto3 not installed. Please run:")
    print("  pip install boto3")
    sys.exit(1)


# Constants
# Maximum number of S3 keys to check when validating architecture
MAX_S3_KEYS_CHECK = 100
BYTES_TO_MB = 1024 * 1024  # Conversion factor from bytes to MB

# Package categories
PACKAGES_TO_PROMOTE = {
    "pytorch_triton_rocm",
    "rocm",
    "rocm_sdk_core",
    "rocm_sdk_devel",
    "rocm_sdk_libraries-*",
    "torch",
    "torchaudio",
    "torchvision",
}

DEPENDENCY_PACKAGES = {
    "filelock",
    "fsspec",
    "jinja2",
    "markupsafe",
    "mpmath",
    "networkx",
    "numpy",
    "pillow",
    "setuptools",
    "sympy",
    "typing_extensions",
}


def categorize_package(filename: str) -> str:
    """Categorize a package file.

    Returns:
        "promote" - package that needs RC version promotion
        "dependency" - dependency package to copy as-is
        "unknown" - unrecognized package
    """
    pkg_name = filename.split("-", 1)[0]

    # Check for rocm-sdk-libraries-* pattern
    if pkg_name.startswith("rocm_sdk_libraries") or pkg_name in PACKAGES_TO_PROMOTE:
        return "promote"

    if pkg_name in DEPENDENCY_PACKAGES:
        return "dependency"

    return "unknown"


def list_architectures(
    s3_client, bucket_name: str, bucket_prefix: str, version: str
) -> List[str]:
    """List all architectures in the bucket that have packages matching the version.

    Args:
        s3_client: boto3 S3 client
        bucket_name: S3 bucket name (e.g., "therock-prerelease-python")
        bucket_prefix: S3 bucket prefix (e.g., "v3/whl/")
        version: Version pattern to search for (e.g., "7.10.0rc2")

    Returns:
        List of architecture names (e.g., ["gfx950-dcgpu", "gfx94X-dcgpu"])
    """
    print(f"Discovering architectures with version {version}...")

    # List all "directories" in the bucket prefix
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=bucket_name, Prefix=bucket_prefix, Delimiter="/"
        )

        architectures = []
        for page in pages:
            if "CommonPrefixes" not in page:
                continue  # Skip empty pages, keep processing
            # "CommonPrefixes" is a list of prefixes that are the "directories" in the S3 bucket
            for prefix in page["CommonPrefixes"]:
                arch = prefix["Prefix"].replace(bucket_prefix, "").rstrip("/")
                # Check if the arch folder contains files with the specified version
                if has_version_in_arch(
                    s3_client, bucket_name, bucket_prefix, arch, version
                ):
                    architectures.append(arch)
                    print(f"  Found: {arch}")

        if len(architectures) == 0:
            print(
                f"""[ERROR]: No architecture subdirectories found in bucket prefix '{bucket_prefix}' of bucket '{bucket_name}'
         that contain packages matching the version '{version}'"""
            )
            sys.exit(1)

        return sorted(architectures)

    except ClientError as e:
        print(f"[ERROR]: Failed to list architectures: {e}")
        sys.exit(1)
    except NoCredentialsError:
        print("[ERROR]: AWS credentials not configured")
        print("Please configure credentials via the IAM role")
        sys.exit(1)


def has_version_in_arch(
    s3_client, bucket_name: str, bucket_prefix: str, arch: str, version: str
) -> bool:
    """Check if an architecture folder contains files with the specified version."""
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=f"{bucket_prefix}{arch}/",
            MaxKeys=MAX_S3_KEYS_CHECK,
        )

        if "Contents" not in response:
            return False

        for obj in response["Contents"]:
            if version in obj["Key"]:
                return True

        return False
    except ClientError:
        return False


def list_packages_for_arch(
    s3_client, bucket_name: str, bucket_prefix: str, arch: str, version: str
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """List all packages for an architecture matching the version.

    Args:
        s3_client: boto3 S3 client
        bucket_name: S3 bucket name
        bucket_prefix: S3 bucket prefix (e.g., "v3/whl/")
        arch: Architecture name (e.g., "gfx950-dcgpu")
        version: Version pattern to filter (e.g., "7.10.0rc2")

    Returns:
        Tuple of (packages_to_promote, dependencies, unknown_packages)
        Each is a list of tuples (full_filepath_in_s3, size)
    """
    prefix = f"{bucket_prefix}{arch}/"

    packages_to_promote = []
    dependencies = []
    unknown = []

    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        for page in pages:
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]
                filename = key.split("/")[-1]

                # Skip directories and index files
                if not filename or filename == "index.html":
                    continue

                # Skip files that don't match version (for packages to promote)
                # Dependencies don't need version matching
                category = categorize_package(filename)

                if category == "promote":
                    if version in filename:
                        packages_to_promote.append((key, obj["Size"]))
                elif category == "dependency":
                    dependencies.append((key, obj["Size"]))
                else:
                    unknown.append((key, obj["Size"]))

        return packages_to_promote, dependencies, unknown

    except ClientError as e:
        print(f"[ERROR]: Failed to list packages for {arch}: {e}")
        return [], [], []


def list_tarball_for_arch(
    s3_client, bucket_name: str, bucket_prefix: str, arch: str, version: str
) -> List[Tuple[str, int]]:
    """List tarball and its size for an architecture matching the version.

    Args:
        s3_client: boto3 S3 client
        bucket_name: S3 bucket name
        bucket_prefix: S3 bucket prefix (e.g., "tarball/")
        arch: Architecture name (e.g., "gfx950-dcgpu")
        version: Version pattern to filter (e.g., "7.10.0rc2")

    Returns:
        List of tuples (tarball_key, size) for matching tarballs
    """
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=bucket_prefix)

        tarballs = []

        for page in pages:
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]
                filename = key.split("/")[-1]

                # Skip directories and index files
                if not filename or filename == "index.html":
                    continue

                if (
                    filename.startswith("therock-dist")
                    and arch + "-" + version in filename
                    and filename.endswith(".tar.gz")
                ):
                    tarballs.append((key, obj["Size"]))

        return tarballs
    except ClientError as e:
        print(f"[ERROR]: Failed to list packages for {arch}: {e}")
        return []


def download_file(s3_client, bucket_name: str, key: str, local_path: Path) -> bool:
    """Download a single file from S3.

    Args:
        s3_client: boto3 S3 client
        bucket_name: S3 bucket name
        key: S3 object key
        local_path: Local file path to save to

    Returns:
        True if successful, False otherwise
    """
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(bucket_name, key, str(local_path))
        return True
    except ClientError as e:
        print(f"    [ERROR]: Failed to download {key}: {e}")
        return False


def download_packages(
    s3_client,
    bucket_name: str,
    bucket_prefix: str,
    arch: str,
    version: str,
    output_dir: Path,
    include_dependencies: bool = False,
) -> Tuple[int, int]:
    """Download packages for an architecture. By default, only packages to promote are downloaded.
       Unknown packages are always skipped.

    Args:
        s3_client: boto3 S3 client
        bucket_name: S3 bucket name
        bucket_prefix: S3 bucket prefix (e.g., "v3/whl/")
        arch: Architecture name
        version: Version pattern
        output_dir: Base output directory
        include_dependencies: Include dependency packages in download (default: False)

    Returns:
        Tuple of (successful_downloads, failed_downloads)
    """
    print(f"\nProcessing architecture: {arch}")
    print("=" * 80)

    packages_to_promote, dependencies, unknown = list_packages_for_arch(
        s3_client, bucket_name, bucket_prefix, arch, version
    )

    print(f"  Packages to promote: {len(packages_to_promote)}")
    print(f"  Dependencies found: {len(dependencies)}")
    if unknown:
        print(f"  Unknown packages (skipped): {len(unknown)}")
        for key in unknown:
            print(f"    - {key[0].split("/")[-1]}")
    print("")
    print("-" * 80)

    arch_dir = output_dir / arch

    success_count = 0
    fail_count = 0

    if include_dependencies:
        all_packages = packages_to_promote + dependencies
        print(
            f"  Downloading {len(all_packages)} packages to promote and their dependencies for {arch} with version {version}..."
        )
    else:
        all_packages = packages_to_promote
        print(
            f"  Downloading {len(all_packages)} python packages to promote for {arch} with version {version}..."
        )

    if not all_packages:
        print(
            f"  [ERROR]: No packages found for {arch} with version {version}. Skipping!"
        )
        return 0, 0

    for idx, obj in enumerate(all_packages):
        key, size = obj
        filename = key.split("/")[-1]
        local_path = arch_dir / filename

        # Skip if already exists
        if local_path.exists():
            print(
                f"  ({idx+1}/{len(all_packages)})   \tSKIP (exists): {filename} ({size/BYTES_TO_MB:.2f} MB)"
            )
            success_count += 1
            continue

        print(
            f"  ({idx+1}/{len(all_packages)})   \tDownloading: {filename} ({size/BYTES_TO_MB:.2f} MB)"
        )
        if download_file(s3_client, bucket_name, key, local_path):
            success_count += 1
        else:
            fail_count += 1

    return success_count, fail_count


def download_tarball(
    s3_client,
    bucket_name: str,
    bucket_prefix: str,
    arch: str,
    version: str,
    output_dir: Path,
) -> Tuple[int, int]:
    """Download tarball for an architecture.

    Args:
        s3_client: boto3 S3 client
        bucket_name: S3 bucket name
        bucket_prefix: S3 bucket prefix
        arch: Architecture name
        version: Version pattern
        output_dir: Base output directory

    Returns:
        Tuple of (successful_download, failed_download)
    """
    tarballs = list_tarball_for_arch(
        s3_client, bucket_name, bucket_prefix, arch, version
    )

    if not tarballs:
        print(
            f"  [ERROR]: No tarball found for {arch} with version {version}. Skipping!"
        )
        return 0, 1

    success_count = 0
    fail_count = 0

    print(f"  Downloading tarball...")

    for name, size in tarballs:
        filename = name.split("/")[-1]
        local_path = output_dir / filename

        # Skip if already exists
        if local_path.exists():
            print(f"        \tSKIP (exists): {filename} ({size/BYTES_TO_MB:.2f} MB)")
            success_count += 1
            continue

        print(f"        \tDownloading: {filename} ({size/BYTES_TO_MB:.2f} MB)")
        if download_file(s3_client, bucket_name, name, local_path):
            success_count += 1
        else:
            fail_count += 1

    return success_count, fail_count


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="Download prerelease packages from S3 for promotion to release",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all architectures for version 7.10.0rc2
  python download_prerelease_packages.py --version=7.10.0rc2 --output-dir=./downloads

  # Download only specific architecture
  python download_prerelease_packages.py --version=7.10.0rc2 --arch=gfx950-dcgpu --output-dir=./downloads

  # Download multiple specific architectures and including tarballs
  python download_prerelease_packages.py --version=7.10.0rc2 --arch=gfx1151,gfx950-dcgpu --output-dir=./downloads --include-tarballs

  # Download packages to promote AND their dependencies
  python download_prerelease_packages.py --version=7.10.0rc2 --output-dir=./downloads --include-dependencies

  # Use custom bucket prefix
  python download_prerelease_packages.py --version=7.10.0rc2 --output-dir=./downloads --bucket-prefix=v3/whl/

  # List available architectures
  python download_prerelease_packages.py --version=7.10.0rc2 --list-archs

  # List all packages per architecture with sizes
  python download_prerelease_packages.py --version=7.10.0rc2 --list-packages-per-arch

  # List packages and tarballs with sizes
  python download_prerelease_packages.py --version=7.10.0rc2 --list-packages-per-arch --include-tarballs
        """,
    )

    parser.add_argument(
        "--version",
        required=True,
        help="Version pattern to download (e.g., '7.10.0rc2')",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for downloaded packages (required unless --list-archs or --list-packages-per-arch)",
    )

    parser.add_argument(
        "--arch",
        help="Download only this specific architecture (e.g. 'gfx950-dcgpu,gfx1151') (default: all architectures)",
    )

    parser.add_argument(
        "--bucket",
        default="therock-prerelease-python",
        help="S3 bucket name (default: therock-prerelease-python)",
    )

    parser.add_argument(
        "--bucket-prefix",
        default="v3/whl/",
        help="S3 bucket prefix for packages (default: v3/whl/)",
    )

    parser.add_argument(
        "--include-dependencies",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include dependency packages in download (default: only packages to promote)",
    )

    parser.add_argument(
        "--include-tarballs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Download also tarball packages (default: False)",
    )
    parser.add_argument(
        "--tarball-bucket",
        default="therock-prerelease-tarball",
        help="S3 bucket name for tarball packages (default: therock-prerelease-tarball)",
    )
    parser.add_argument(
        "--tarball-bucket-prefix",
        default="v3/tarball/",
        help="S3 bucket prefix for tarball packages (default: v3/tarball/)",
    )
    parser.add_argument(
        "--tarball-output-dir",
        type=Path,
        help="Output directory for downloaded tarball packages (default: < --output-dir >/tarball)",
    )

    parser.add_argument(
        "--list-archs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only list available architectures, do not download",
    )

    parser.add_argument(
        "--list-packages-per-arch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="List all packages per architecture, do not download",
    )

    args = parser.parse_args(argv)

    if args.arch:
        args.arch = [arch.strip() for arch in args.arch.split(",")]

    # Validate arguments
    if args.include_tarballs:
        if args.tarball_output_dir is None and args.output_dir is not None:
            args.tarball_output_dir = args.output_dir / "tarball"
        else:
            args.tarball_output_dir = (
                Path(args.tarball_output_dir)
                if args.tarball_output_dir is not None
                else None
            )
    if not args.list_archs and not args.list_packages_per_arch and not args.output_dir:
        parser.error(
            "--output-dir is required unless --list-archs or --list-packages-per-arch is specified"
        )

    return args


def print_packages_per_arch(
    s3_client,
    bucket_name: str,
    bucket_prefix: str,
    architectures: List[str],
    version: str,
    include_tarballs: bool = False,
    tarball_bucket_name: str = None,
    tarball_bucket_prefix: str = None,
) -> Dict[str, Dict[str, List[str]]]:
    print(
        "\n--list-packages-per-arch specified, listing packages and their sizes without download"
    )
    print("=" * 80)

    all_packages = {}
    total_size_promote = 0
    total_size_deps = 0
    total_size_unknown = 0
    total_size_tarball = 0
    for arch in architectures:
        print(f"\n{arch}:")
        print("-" * 80)
        packages_to_promote, dependencies, unknown = list_packages_for_arch(
            s3_client, bucket_name, bucket_prefix, arch, version
        )

        all_packages[arch] = {
            "packages_to_promote": packages_to_promote,
            "dependencies": dependencies,
            "unknown": unknown,
        }

        size_promote = 0
        size_dependencies = 0
        size_unknown = 0
        size_tarball = 0
        print(f"  Packages to promote ({len(packages_to_promote)}):")
        for pkg in all_packages[arch]["packages_to_promote"]:
            name, size = pkg
            name = name.split("/")[-1]
            size_promote += size
            total_size_promote += size
            print(f"    - {name} ({size/BYTES_TO_MB:.2f} MB)")

        print(f"\n  Dependencies found ({len(dependencies)}):")
        for pkg in all_packages[arch]["dependencies"]:
            name, size = pkg
            name = name.split("/")[-1]
            size_dependencies += size
            total_size_deps += size
            print(f"    - {name} ({size/BYTES_TO_MB:.2f} MB)")

        if unknown:
            print(f"\n  Unknown packages ({len(unknown)}):")
            for pkg in all_packages[arch]["unknown"]:
                name, size = pkg
                name = name.split("/")[-1]
                size_unknown += size
                total_size_unknown += size
                print(f"    - {name} ({size/BYTES_TO_MB:.2f} MB)")

        if include_tarballs:
            print(f"\n  Tarball:")
            tarballs = list_tarball_for_arch(
                s3_client, tarball_bucket_name, tarball_bucket_prefix, arch, version
            )
            for tarball in tarballs:
                tarball_name, tarball_size = tarball
                size_tarball += tarball_size
                total_size_tarball += tarball_size
                size_mb = tarball_size / BYTES_TO_MB
                print(f"    - {tarball_name.split("/")[-1]} ({size_mb:.2f} MB)")
            if not tarballs:
                print(
                    f"  [WARN]: No tarball found for {arch} with version {version}. Skipping!"
                )

        print("")
        print("-" * 80)
        print("")
        print(f"  Space needed for {arch}:")
        print(f"    Packages to promote:   {size_promote/BYTES_TO_MB:>12.2f} MB")
        print(f"    Dependencies packages: {size_dependencies/BYTES_TO_MB:>12.2f} MB")
        print(f"    Unknown packages:      {size_unknown/BYTES_TO_MB:>12.2f} MB")
        print(f"    Tarball:               {size_tarball/BYTES_TO_MB:>12.2f} MB")
        print("")
        print("-" * 80)
        print("")

    print("")
    print("-" * 80)
    print("")
    print("  Total space needed for all architectures:")
    print(f"    All packages to promote:    {total_size_promote/BYTES_TO_MB:>12.2f} MB")
    print(f"    All dependencies packages:  {total_size_deps/BYTES_TO_MB:>12.2f} MB")
    print(f"    All unknown packages:       {total_size_unknown/BYTES_TO_MB:>12.2f} MB")
    print(f"    All tarball:                {total_size_tarball/BYTES_TO_MB:>12.2f} MB")
    print("")
    if include_tarballs:
        print(
            f"    All packages to promote and tarballs:  {(total_size_promote + total_size_tarball)/BYTES_TO_MB:>12.2f} MB"
        )
    print("-" * 80)
    print("")

    return all_packages


def download_prerelease_packages(
    version: str,
    output_dir: Path = None,
    architectures: List[str] = None,
    bucket_name: str = "therock-prerelease-python",
    bucket_prefix: str = "v3/whl/",
    include_dependencies: bool = False,
    include_tarballs: bool = False,
    tarball_bucket_name: str = "therock-prerelease-tarball",
    tarball_bucket_prefix: str = "v3/tarball/",
    tarball_output_dir: Path = None,
    list_archs: bool = False,
    list_packages_per_arch: bool = False,
) -> Union[List[str], Dict[str, Dict[str, List[str]]], Tuple[int, int, List[str]]]:
    """Download prerelease packages from S3 bucket for promotion to release.

    Args:
        version: Version pattern to download (e.g., '7.10.0rc2')
        output_dir: Output directory for downloaded packages (required unless list_archs or list_packages_per_arch)
        architectures: Download only these specific architectures (default: all architectures)
        bucket_name: S3 bucket name (default: therock-prerelease-python)
        bucket_prefix: S3 bucket prefix for packages (default: v3/whl/)
        include_dependencies: Include dependency packages in download (default: False).
                              Ignored if list_archs or list_packages_per_arch is True.
        include_tarballs: Include tarballs in download and listings (default: False).
        tarball_bucket_name: S3 bucket name for tarball packages (default: therock-prerelease-tarball)
        tarball_bucket_prefix: S3 bucket prefix for tarball packages (default: v3/tarball/)
        tarball_output_dir: Output directory for downloaded tarball packages (default: < --output-dir >/tarballs)
        list_archs: Only list available architectures, do not download (default: False).
                   Set by CLI flag --list-archs
        list_packages_per_arch: List all packages per architecture and their sizes, do not download (default: False)

    Returns:
        If list_archs=True: List of architecture names
        If list_packages_per_arch=True: Dict mapping arch to dict of package categories
        Otherwise: Tuple of (total_success, total_fail, architectures) of downloaded packages

    Raises:
        SystemExit: If AWS credentials are not configured, no architectures found, or downloads fail
    """
    # Validate arguments
    if not list_archs and not list_packages_per_arch and output_dir is None:
        print(
            "[ERROR]: output_dir is required unless list_archs=True or list_packages_per_arch=True"
        )
        sys.exit(1)

    print("=" * 80)
    print("Download Prerelease Packages")
    print("=" * 80)
    print(f"Bucket: {bucket_name}")
    print(f"Version: {version}")
    if architectures:
        print(f"Architectures: {architectures} (user-specified)")
    else:
        print(f"Architecture: ALL")
    print("=" * 80)

    s3_client = boto3.client("s3")
    # List architectures
    if architectures:

        # Validate they exist
        for arch in architectures:
            if not has_version_in_arch(
                s3_client, bucket_name, bucket_prefix, arch, version
            ):
                print(
                    f"[ERROR]: Architecture {arch} not found or has no packages with version {version}"
                )
                sys.exit(1)
        print(f"\nVerified user-specified architectures exist: {architectures}")
    else:
        architectures = list_architectures(
            s3_client, bucket_name, bucket_prefix, version
        )

        print(f"\nFound {len(architectures)} architecture(s)")

    if list_archs:
        print("\n--list-archs specified, exiting without download")
        return architectures

    if list_packages_per_arch:
        return print_packages_per_arch(
            s3_client=s3_client,
            bucket_name=bucket_name,
            bucket_prefix=bucket_prefix,
            architectures=architectures,
            version=version,
            include_tarballs=include_tarballs,
            tarball_bucket_name=tarball_bucket_name,
            tarball_bucket_prefix=tarball_bucket_prefix,
        )
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir.absolute()}")

    if include_tarballs:
        tarball_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Tarball output directory: {tarball_output_dir.absolute()}")

    # Download packages for each architecture
    total_success = 0
    total_fail = 0

    for arch in architectures:
        success_count = 0
        fail_count = 0
        success, fail = download_packages(
            s3_client,
            bucket_name,
            bucket_prefix,
            arch,
            version,
            output_dir,
            include_dependencies,
        )
        success_count += success
        fail_count += fail
        total_success += success
        total_fail += fail

        if include_tarballs:
            success, fail = download_tarball(
                s3_client,
                tarball_bucket_name,
                tarball_bucket_prefix,
                arch,
                version,
                tarball_output_dir,
            )
            success_count += success
            fail_count += fail
            total_success += success
            total_fail += fail

        print(f"\n  Summary for {arch}:")
        print(f"    Successful: {success_count}")
        print(f"    Failed: {fail_count}")

    # Final summary
    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"Total architectures: {len(architectures)}")
    print(f"Total successful downloads: {total_success}")
    print(f"Total failed downloads: {total_fail}")

    print(f"\nOutput directory: {output_dir.absolute()}/<arch>")
    if include_tarballs:
        print(f"Tarball output directory: {tarball_output_dir.absolute()}")

    print(f"\nArchitectures downloaded: {architectures}")

    if total_fail > 0:
        print("\nWARNING: Some downloads failed!")
        sys.exit(1)

    print("\nFor next steps check: how_to_do_release.md")
    print("=" * 80)

    return total_success, total_fail, architectures


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])

    download_prerelease_packages(
        version=args.version,
        output_dir=args.output_dir,
        architectures=args.arch,
        bucket_name=args.bucket,
        bucket_prefix=args.bucket_prefix,
        include_dependencies=args.include_dependencies,
        include_tarballs=args.include_tarballs,
        tarball_bucket_name=args.tarball_bucket,
        tarball_bucket_prefix=args.tarball_bucket_prefix,
        tarball_output_dir=args.tarball_output_dir,
        list_archs=args.list_archs,
        list_packages_per_arch=args.list_packages_per_arch,
    )
