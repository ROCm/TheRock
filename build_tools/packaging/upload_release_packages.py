#!/usr/bin/env python
"""Upload promoted packages to S3 release bucket.

This script uploads promoted packages to S3 release buckets. It uploads wheel
files from architecture subdirectories and optionally tarball files. It expects
the same directory layout as download_prerelease_packages.py creates.

SAFETY FEATURES:
  - DRY-RUN by default: Shows what would be uploaded without actually uploading
  - Requires --execute flag to perform actual S3 uploads
  - Testing buckets by default (you can use --use-release-buckets for production)

PREREQUISITES:
  - pip install -r ./build_tools/packaging/requirements.txt
  - AWS credentials configured (IAM role)
  - Packages must be promoted first using promote_from_rc_to_final.py

TYPICAL USAGE (Command Line):
  # Preview wheel upload (dry-run, no actual upload)
  python ./build_tools/packaging/upload_release_packages.py \
    --input-dir=./downloads

  # Upload wheels only (to testing bucket)
  python ./build_tools/packaging/upload_release_packages.py \
    --input-dir=./downloads \
    --execute

  # Upload wheels and tarballs to production release buckets
  python ./build_tools/packaging/upload_release_packages.py \
    --input-dir=./downloads \
    --upload-tarballs \
    --execute \
    --use-release-buckets

DIRECTORY STRUCTURE:
  Input directory structure (created by download_prerelease_packages.py):
    <input-dir>/
      <arch1>/
        package1.whl
        package2.whl
        ...
      <arch2>/
        ...
      tarball/  (optional, if --include-tarballs was used)
        therock-dist-linux-<arch1>-<version>.tar.gz
        therock-dist-windows-<arch2>-<version>.tar.gz
        ...

  S3 bucket structure:
    v3/whl/<arch>/
      package1.whl
      package2.whl
      ...
    v3/tarball/
      therock-dist-linux-<arch1>-<version>.tar.gz
      therock-dist-windows-<arch2>-<version>.tar.gz
      ...

NOTE:
  Index file generation is handled separately by manage.py.
  This script only uploads package files.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    print("[ERROR]: boto3 not installed. Please run:")
    print("  pip install boto3")
    sys.exit(1)


def upload_python_files(
    input_dir: Path, bucket_name: str, bucket_prefix: str, execute: bool
) -> int:
    """Upload Python package files to S3 bucket.

    Args:
        input_dir: Root directory containing architecture subdirectories
        bucket_name: S3 bucket name
        bucket_prefix: S3 bucket prefix (e.g., 'v3/whl/')
        execute: If False, only show what would be uploaded (dry-run)

    Returns:
        Number of files uploaded (or would be uploaded)
    """
    print("\nUploading Python packages to S3...")
    print("=" * 80)

    if not execute:
        print("DRY-RUN MODE: No actual uploads will be performed")
        print("=" * 80)

    s3_client = boto3.client("s3") if execute else None
    upload_count = 0

    for arch_dir in input_dir.iterdir():
        if not arch_dir.is_dir():
            continue

        # Skip tarball directory (handled separately)
        if arch_dir.name == "tarball":
            continue

        arch = arch_dir.name
        print(f"\nArchitecture: {arch}")

        # Find all wheel and tar.gz files (rocm sdist)
        files_to_upload = []
        files_to_upload.extend(arch_dir.glob("*.whl"))
        files_to_upload.extend(arch_dir.glob("*.tar.gz"))

        if not files_to_upload:
            print(f"  No files to upload")
            continue

        print(f"  Found {len(files_to_upload)} file(s) to upload:")

        for file_path in sorted(files_to_upload):
            s3_key = f"{bucket_prefix}{arch}/{file_path.name}"

            if execute:
                try:
                    print(f"    Uploading {file_path.name} ...", end="", flush=True)
                    s3_client.upload_file(str(file_path), bucket_name, s3_key)
                    print(" done")
                    upload_count += 1
                except ClientError as e:
                    print(
                        f"\n    [ERROR]: Could not upload {file_path.name} to s3://{bucket_name}/{s3_key}: {e}"
                    )
            else:
                print(
                    f"    [DRY-RUN] Would upload: {file_path.name} -> s3://{bucket_name}/{s3_key}"
                )
                upload_count += 1

    return upload_count


def upload_tarball_files(
    input_dir: Path, bucket_name: str, bucket_prefix: str, execute: bool
) -> int:
    """Upload tarball files to S3 bucket.

    Args:
        input_dir: Root directory containing tarball subdirectory
        bucket_name: S3 bucket name
        bucket_prefix: S3 bucket prefix (e.g., 'v3/tarball/')
        execute: If False, only show what would be uploaded (dry-run)

    Returns:
        Number of tarball files uploaded (or would be uploaded)
    """
    print("\nUploading tarballs to S3...")
    print("=" * 80)

    if not execute:
        print("DRY-RUN MODE: No actual uploads will be performed")
        print("=" * 80)

    tarball_dir = input_dir / "tarball"

    if not tarball_dir.exists() or not tarball_dir.is_dir():
        print(f"  No tarball directory found at {tarball_dir}")
        return 0

    s3_client = boto3.client("s3") if execute else None
    upload_count = 0

    # Find all tar.gz files
    tarball_files = list(tarball_dir.glob("*.tar.gz"))

    if not tarball_files:
        print(f"  No tarball files found")
        return 0

    print(f"  Found {len(tarball_files)} tarball file(s) to upload:")

    for file_path in sorted(tarball_files):
        s3_key = f"{bucket_prefix}{file_path.name}"

        if execute:
            try:
                print(f"    Uploading {file_path.name} ...", end="", flush=True)
                s3_client.upload_file(str(file_path), bucket_name, s3_key)
                print(" done")
                upload_count += 1
            except ClientError as e:
                print(
                    f"\n    [ERROR]: Could not upload {file_path.name} to s3://{bucket_name}/{s3_key}: {e}"
                )
        else:
            print(
                f"    [DRY-RUN] Would upload: {file_path.name} -> s3://{bucket_name}/{s3_key}"
            )
            upload_count += 1

    return upload_count


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="Upload promoted packages to S3 release bucket",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run: Preview wheel upload (DEFAULT - no actual uploads)
  python upload_release_packages.py --input-dir=./downloads

  # Upload wheels to test bucket
  python upload_release_packages.py --input-dir=./downloads --execute

  # Upload wheels and tarballs to production release buckets
  python upload_release_packages.py --input-dir=./downloads --upload-tarballs --execute --use-release-buckets

  # Upload only tarballs to production
  python upload_release_packages.py --input-dir=./downloads --no-upload-python --upload-tarballs --execute --use-release-buckets

Safety Features:
  - Dry-run is the DEFAULT mode (no --execute flag = no uploads)
  - Testing buckets by DEFAULT (requires --use-release-buckets for production)
  - Use --execute flag to perform actual uploads
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory containing architecture subdirectories with promoted packages",
    )

    parser.add_argument(
        "--upload-python",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload Python packages (default: True)",
    )

    parser.add_argument(
        "--bucket",
        default="therock-testing-bucket",
        help="S3 bucket name for Python packages (default: therock-testing-bucket). You probably want to use therock-release-python",
    )

    parser.add_argument(
        "--bucket-prefix",
        default="release-upload-testing/",
        help="S3 bucket prefix for Python packages (default: release-upload-testing/)",
    )

    parser.add_argument(
        "--upload-tarballs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload tarball packages (default: False)",
    )

    parser.add_argument(
        "--tarball-bucket",
        default="therock-testing-bucket",
        help="S3 bucket name for tarball packages (default: therock-testing-bucket). You probably want to use therock-release-tarball",
    )

    parser.add_argument(
        "--tarball-bucket-prefix",
        default="release-upload-testing/tarball/",
        help="S3 bucket prefix for tarball packages (default: release-upload-testing/tarball/)",
    )

    parser.add_argument(
        "--execute",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Actually perform uploads (default: dry-run mode)",
    )

    parser.add_argument(
        "--use-release-buckets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite buckets to use production release buckets: therock-release-tarball:v3/tarball and therock-release-python:v3/whl",
    )

    args = parser.parse_args(argv)

    if args.use_release_buckets:
        args.bucket = "therock-release-python"
        args.bucket_prefix = "v3/whl/"
        args.tarball_bucket = "therock-release-tarball"
        args.tarball_bucket_prefix = "v3/tarball/"

    # Validate input directory
    if not args.input_dir.exists():
        parser.error(f"Input directory does not exist: {args.input_dir}")

    if not args.input_dir.is_dir():
        parser.error(f"Input path is not a directory: {args.input_dir}")

    return args


def upload_release_packages(
    input_dir: Path,
    upload_python: bool = True,
    bucket_name: str = "therock-testing-bucket",
    bucket_prefix: str = "release-upload-testing/",
    upload_tarballs: bool = False,
    tarball_bucket_name: str = "therock-testing-bucket",
    tarball_bucket_prefix: str = "release-upload-testing/tarball/",
    execute: bool = False,
) -> Tuple[int, int]:
    """Upload promoted packages to S3 release bucket.

    Args:
        input_dir: Root directory containing architecture subdirectories with promoted packages
        upload_python: Upload Python packages (default: True)
        bucket_name: S3 bucket name for Python packages (default: therock-testing-bucket)
        bucket_prefix: S3 bucket prefix for Python packages (default: release-upload-testing/)
        upload_tarballs: Upload tarball packages (default: False)
        tarball_bucket_name: S3 bucket name for tarball packages (default: therock-testing-bucket)
        tarball_bucket_prefix: S3 bucket prefix for tarball packages (default: release-upload-testing/tarball/)
        execute: If True, perform actual uploads; if False, dry-run mode (default: False)

    Returns:
        Tuple of (python_packages_uploaded, tarballs_uploaded)

    Raises:
        SystemExit: If AWS credentials are not configured or S3 operations fail
    """
    print("=" * 80)
    print("Upload Release Packages")
    print("=" * 80)
    print(f"Input directory: {input_dir.absolute()}")
    print(f"Python bucket: {bucket_name}")
    print(f"Python bucket prefix: {bucket_prefix}")
    if upload_tarballs:
        print(f"Tarball bucket: {tarball_bucket_name}")
        print(f"Tarball bucket prefix: {tarball_bucket_prefix}")
    print(
        f"Execution mode: {'EXECUTE (will upload)' if execute else 'DRY-RUN (no uploads)'}"
    )
    print("=" * 80)

    if not execute:
        print("\n⚠️  SAFETY: Dry-run mode active - no actual uploads will be performed")
        print("⚠️          Use --execute flag to perform actual uploads")
        print(
            "⚠️          Additionally use --use-release-buckets flag for production buckets (therock-release-python and therock-release-tarball)\n"
        )

    # Count architectures (exclude tarball directory)
    architectures = [
        d for d in input_dir.iterdir() if d.is_dir() and d.name != "tarball"
    ]
    print(f"\nFound {len(architectures)} architecture(s):")
    for arch_dir in architectures:
        print(f"  - {arch_dir.name}")

    if not architectures:
        print("\n[ERROR]: No architecture directories found in input directory")
        sys.exit(1)

    # Check if at least one upload type is enabled
    if not upload_python and not upload_tarballs:
        print(
            "\n[ERROR]: No upload targets specified. Use --upload-python and/or --upload-tarballs"
        )
        sys.exit(1)

    # Upload packages
    wheels_uploaded = 0
    tarballs_uploaded = 0

    try:
        if upload_python:
            wheels_uploaded = upload_python_files(
                input_dir, bucket_name, bucket_prefix, execute
            )

        if upload_tarballs:
            tarballs_uploaded = upload_tarball_files(
                input_dir, tarball_bucket_name, tarball_bucket_prefix, execute
            )

    except NoCredentialsError:
        print("\n[ERROR]: AWS credentials not configured")
        print("Please configure credentials via IAM role")
        sys.exit(1)
    except ClientError as e:
        print(f"\n[ERROR]: S3 operation failed: {e}")
        sys.exit(1)

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if execute:
        print("Execution mode: EXECUTED")
        if wheels_uploaded > 0:
            print(
                f"\nPython packages uploaded: {wheels_uploaded} (location: s3://{bucket_name}/{bucket_prefix})"
            )
        if tarballs_uploaded > 0:
            print(
                f"Tarballs uploaded: {tarballs_uploaded} (location: s3://{tarball_bucket_name}/{tarball_bucket_prefix})"
            )
    else:
        print("Execution mode: DRY-RUN")
        if wheels_uploaded > 0:
            print(f"  Python packages that would be uploaded: {wheels_uploaded}")
        if tarballs_uploaded > 0:
            print(f"  Tarballs that would be uploaded: {tarballs_uploaded}")

    print("\nNext steps:")
    if execute:
        print("  - Verify uploads in S3 bucket")
        print("  - Use manage.py to generate and upload index files")
    else:
        print("  - Review the dry-run output above")
        print(
            "  - Run again with --execute to perform actual uploads to testing bucket"
        )
        print("  - Add --use-release-buckets flag when ready for production upload")

    print("=" * 80)

    return wheels_uploaded, tarballs_uploaded


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    upload_release_packages(
        input_dir=args.input_dir,
        upload_python=args.upload_python,
        bucket_name=args.bucket,
        bucket_prefix=args.bucket_prefix,
        upload_tarballs=args.upload_tarballs,
        tarball_bucket_name=args.tarball_bucket,
        tarball_bucket_prefix=args.tarball_bucket_prefix,
        execute=args.execute,
    )
