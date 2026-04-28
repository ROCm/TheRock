#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate public download URLs for uploaded tarballs.

This resolves the uploaded tarball destination using WorkflowOutputRoot,
verifies the expected tarball objects exist in S3, and writes public
download URLs to GitHub Actions outputs.

Outputs written to GITHUB_OUTPUT:
    tarball_url
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import boto3

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from _therock_utils.workflow_outputs import WorkflowOutputRoot
from github_actions_api import gha_set_output


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate public download URLs for uploaded tarballs"
    )
    parser.add_argument("--run-id", required=True, help="Workflow run ID")
    parser.add_argument(
        "--platform",
        required=True,
        choices=["linux", "windows"],
        help="Platform for workflow outputs",
    )
    parser.add_argument(
        "--release-type",
        default="",
        help='Release type: "" for CI, or "dev", "nightly", "prerelease"',
    )
    parser.add_argument(
        "--package-version",
        required=True,
        help="ROCm/TheRock package version used in tarball names",
    )
    parser.add_argument(
        "--dist-amdgpu-families",
        required=True,
        help="Semicolon-separated family list used for tarball generation",
    )
    parser.add_argument(
        "--family",
        required=True,
        help="AMDGPU family to generate a tarball URL for",
    )
    parser.add_argument(
        "--aws-region",
        default="us-east-2",
        help="AWS region for S3 client",
    )
    return parser.parse_args(argv)


def parse_family_list(raw: str) -> list[str]:
    return [name.strip() for name in raw.split(";") if name.strip()]


def get_tarball_name(
    *,
    platform: str,
    package_version: str,
    family: str,
) -> str:
    return f"therock-dist-{platform}-{family}-{package_version}.tar.gz"


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Unexpected S3 URI: {s3_uri}")
    bucket_and_key = s3_uri[len("s3://") :]
    bucket, prefix = bucket_and_key.split("/", 1)
    return bucket, prefix


def generate_public_s3_url(
    *,
    s3_client,
    bucket: str,
    key: str,
) -> str:
    # Fail fast if the uploaded object is not present where we expect it.
    s3_client.head_object(Bucket=bucket, Key=key)
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    families = parse_family_list(args.dist_amdgpu_families)

    output_root = WorkflowOutputRoot.from_workflow_run(
        run_id=args.run_id,
        platform=args.platform,
        release_type=args.release_type or None,
    )
    dest = output_root.tarballs()
    bucket, prefix = parse_s3_uri(dest.s3_uri)

    if args.family not in families:
        raise ValueError(
            f"Requested family '{args.family}' is not in dist families: {families}"
        )

    tarball_name = get_tarball_name(
        platform=args.platform,
        package_version=args.package_version,
        family=args.family,
    )

    s3 = boto3.client("s3", region_name=args.aws_region)

    tarball_key = f"{prefix}/{tarball_name}"
    tarball_url = generate_public_s3_url(
        s3_client=s3,
        bucket=bucket,
        key=tarball_key,
    )

    gha_set_output({"tarball_url": tarball_url})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
