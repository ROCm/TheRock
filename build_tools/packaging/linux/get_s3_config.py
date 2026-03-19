#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Determine S3 bucket and prefix for ROCm package uploads.

This script implements the S3 bucket selection logic for native Linux packages,
determining the appropriate bucket and prefix based on release type, repository,
and fork status.

Decision Tree:
  ├─ IF release_type is set (dev/nightly/prerelease/release)
  │  └─ Use: therock-${release_type}-packages bucket
  │     ├─ prerelease/release → prefix: v3/packages/<pkg_type>
  │     └─ dev/nightly → prefix: v3/packages/<pkg_type>/<YYYYMMDD>-<artifact_id>
  │
  ├─ ELSE IF fork PR OR non-ROCm/TheRock repository
  │  └─ Use: therock-ci-artifacts-external bucket (external CI)
  │     └─ prefix: v3/packages/<pkg_type>/<YYYYMMDD>-<artifact_id>
  │
  └─ ELSE (default: ROCm/TheRock non-fork)
     └─ Use: therock-ci-artifacts bucket (internal CI)
        └─ prefix: v3/packages/<pkg_type>/<YYYYMMDD>-<artifact_id>

Usage:
    # GitHub Actions output format
    python get_s3_config.py \\
        --release-type dev \\
        --repository "ROCm/TheRock" \\
        --is-fork false \\
        --pkg-type deb \\
        --artifact-id 12345678 \\
        --output-format github

    # Environment variables format
    python get_s3_config.py \\
        --release-type "" \\
        --repository "ROCm/TheRock" \\
        --is-fork false \\
        --pkg-type rpm \\
        --artifact-id 12345678 \\
        --output-format env

    # JSON format
    python get_s3_config.py \\
        --release-type nightly \\
        --repository "ROCm/TheRock" \\
        --is-fork false \\
        --pkg-type deb \\
        --artifact-id 12345678 \\
        --output-format json
"""

import argparse
import json
import re
import sys
from datetime import datetime
from typing import Optional, Tuple


def extract_date_from_version(version: Optional[str]) -> str:
    """
    Extract date from ROCm package version string.

    Supports various version formats:
    - Debian dev:     8.1.0~dev20251203      → 20251203
    - Debian nightly: 8.1.0~20251203         → 20251203
    - RPM:            8.1.0~20251203gf689a8e → 20251203
    - Wheel/alpha:    7.10.0a20251021        → 20251021

    Falls back to current date if no date is found in the version string.

    Args:
        version: ROCm package version string (may be None)

    Returns:
        Date string in YYYYMMDD format
    """
    if not version:
        return datetime.now().strftime("%Y%m%d")

    # Look for 8-digit date pattern (YYYYMMDD)
    # Common patterns: ~dev20251203, ~20251203, a20251021, ~20251203gf689a8e
    match = re.search(r"(\d{8})", version)
    if match:
        return match.group(1)

    # No date found, fall back to current date
    return datetime.now().strftime("%Y%m%d")


def determine_s3_config(
    release_type: str,
    repository: str,
    is_fork: bool,
    pkg_type: str,
    artifact_id: str,
    rocm_version: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    Determine S3 bucket, prefix, and job type based on inputs.

    Args:
        release_type: Release type ('dev', 'nightly', 'prerelease', 'release', 'ci', or empty)
        repository: GitHub repository name (e.g., 'ROCm/TheRock')
        is_fork: Whether this is a fork PR
        pkg_type: Package type ('deb' or 'rpm')
        artifact_id: Artifact/run ID for versioning
        rocm_version: ROCm package version string (for date extraction)

    Returns:
        Tuple of (s3_bucket, s3_prefix, job_type)
    """
    # Extract date from version for consistency between version and S3 path
    yyyymmdd = extract_date_from_version(rocm_version)

    # Branch 1: Release-type-specific package buckets (dev/nightly/prerelease/release)
    # Note: 'ci' or empty string should fall through to CI bucket logic below
    if release_type and release_type not in ("", "ci"):
        s3_bucket = f"therock-{release_type}-packages"

        if release_type in ("prerelease", "release"):
            # Prerelease/Release packages go to stable prefix (no date subfolder)
            s3_prefix = f"v3/packages/{pkg_type}"
            job_type = release_type
            print(f"✓ Using release-type bucket: {s3_bucket}", file=sys.stderr)
        else:
            # Dev/Nightly packages go to dated subfolder for versioning
            s3_prefix = f"v3/packages/{pkg_type}/{yyyymmdd}-{artifact_id}"
            job_type = release_type
            print(f"✓ Using release-type bucket: {s3_bucket}", file=sys.stderr)

    # Branch 2: Fork PRs or external repositories
    elif is_fork or repository != "ROCm/TheRock":
        s3_bucket = "therock-ci-artifacts-external"
        s3_prefix = f"v3/packages/{pkg_type}/{yyyymmdd}-{artifact_id}"
        job_type = "ci"
        print(f"✓ Using external bucket: {s3_bucket}", file=sys.stderr)

    # Branch 3: Default - ROCm/TheRock non-fork (normal CI builds)
    else:
        s3_bucket = "therock-ci-artifacts"
        s3_prefix = f"v3/packages/{pkg_type}/{yyyymmdd}-{artifact_id}"
        job_type = "ci"
        print(f"✓ Using default CI bucket: {s3_bucket}", file=sys.stderr)

    print(f"S3 bucket: {s3_bucket}", file=sys.stderr)
    print(f"S3 prefix: {s3_prefix}", file=sys.stderr)
    print(f"Job type: {job_type}", file=sys.stderr)

    return s3_bucket, s3_prefix, job_type


def main():
    parser = argparse.ArgumentParser(
        description="Determine S3 bucket and prefix for ROCm package uploads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--release-type",
        required=True,
        help="Release type ('dev', 'nightly', 'prerelease', 'release', 'ci', or empty string for CI builds)",
    )
    parser.add_argument(
        "--repository",
        required=True,
        help="GitHub repository name (e.g., 'ROCm/TheRock')",
    )
    parser.add_argument(
        "--is-fork",
        required=True,
        help="Whether this is a fork PR ('true' or 'false')",
    )
    parser.add_argument(
        "--pkg-type",
        required=True,
        choices=["deb", "rpm"],
        help="Package type",
    )
    parser.add_argument(
        "--artifact-id",
        required=True,
        help="Artifact/run ID for versioning",
    )
    parser.add_argument(
        "--rocm-version",
        required=False,
        default=None,
        help="ROCm package version string (for date extraction, e.g., '8.1.0~dev20251203')",
    )
    parser.add_argument(
        "--output-format",
        choices=["env", "json", "github"],
        default="env",
        help="Output format: 'env' for shell variables, 'json' for JSON, 'github' for GitHub Actions",
    )

    args = parser.parse_args()

    # Convert string to boolean
    is_fork = args.is_fork.lower() in ("true", "1", "yes")

    # Determine S3 configuration
    s3_bucket, s3_prefix, job_type = determine_s3_config(
        release_type=args.release_type,
        repository=args.repository,
        is_fork=is_fork,
        pkg_type=args.pkg_type,
        artifact_id=args.artifact_id,
        rocm_version=args.rocm_version,
    )

    # Output in requested format
    if args.output_format == "json":
        output = {
            "s3_bucket": s3_bucket,
            "s3_prefix": s3_prefix,
            "job_type": job_type,
        }
        print(json.dumps(output, indent=2))
    elif args.output_format == "github":
        # GitHub Actions output format
        print(f"s3_bucket={s3_bucket}")
        print(f"s3_prefix={s3_prefix}")
        print(f"job_type={job_type}")
    else:  # env format
        print(f"export S3_BUCKET={s3_bucket}")
        print(f"export S3_PREFIX={s3_prefix}")
        print(f"export JOB_TYPE={job_type}")


if __name__ == "__main__":
    main()
