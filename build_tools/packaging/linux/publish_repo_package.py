#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Publish a built amdrocm-repo package to S3 as a standalone per-distro file.

The package is uploaded next to the native content packages, as a
``repo/<os-profile>/`` sibling of the per-run ``packages/<format>/`` prefix,
rather than into the repository index. A client can then fetch it directly by
URL to configure the repository. The file becomes publicly reachable once the
release promotion step copies the per-run tree to the public CDN.

The object name is fixed (``amdrocm-repo.<ext>``) so the download URL is stable
regardless of the built package's versioned filename.

Usage:
  python build_tools/packaging/linux/publish_repo_package.py \
      --file repo-package-out/amdrocm-repo-7.14.0-1.el10.noarch.rpm \
      --bucket therock-prerelease-artifacts \
      --prefix 12345-linux/packages/rpm \
      --os-profile rhel10 \
      --pkg-type rpm
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# Fixed published object name; the extension is the package type.
PUBLISHED_NAME = "amdrocm-repo"

# An os-profile becomes a path segment in the S3 key, so restrict it to a safe
# set of characters (no slashes, whitespace, or control characters). \Z (not $)
# so a trailing newline is not accepted.
_OS_PROFILE_RE = re.compile(r"^[A-Za-z0-9._-]+\Z")


def object_key(prefix: str, os_profile: str, pkg_type: str) -> str:
    """Return the S3 key for the published bootstrap file.

    ``{prefix}/repo/{os_profile}/amdrocm-repo.{ext}`` — a sibling of the content
    repository under the same per-run prefix, so it is not part of the repo
    index (``dists/`` for deb, ``x86_64/repodata`` for rpm).
    """
    return f"{prefix.rstrip('/')}/repo/{os_profile}/{PUBLISHED_NAME}.{pkg_type}"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Publish an amdrocm-repo package to S3.",
    )
    p.add_argument("--file", required=True, type=Path, help="Built package to upload")
    p.add_argument("--bucket", required=True, help="Destination S3 bucket")
    p.add_argument(
        "--prefix",
        required=True,
        help="Per-run S3 prefix, e.g. <run_id>-linux/packages/<format>",
    )
    p.add_argument(
        "--os-profile", required=True, help="Target distro profile (path segment)"
    )
    p.add_argument(
        "--pkg-type",
        required=True,
        choices=["deb", "rpm"],
        help="Package type, used as the published file extension",
    )
    p.add_argument(
        "--endpoint-url",
        default=None,
        help="S3 endpoint override (for testing against a local server)",
    )
    args = p.parse_args(argv)
    # Fail loudly rather than silently uploading nowhere if the bucket is empty
    # (e.g. an unresolved release line without upload credentials).
    if not args.bucket.strip():
        p.error("--bucket must be a non-empty bucket name")
    if not _OS_PROFILE_RE.match(args.os_profile):
        p.error(f"--os-profile has an unexpected format: {args.os_profile!r}")
    if not args.file.is_file():
        p.error(f"--file is not a file: {args.file}")
    return args


def publish(args: argparse.Namespace) -> str:
    """Upload the file and confirm it landed. Returns the object key."""
    # Deferred import: boto3 is only needed for the actual upload, so key
    # derivation stays importable without it.
    import boto3

    key = object_key(args.prefix, args.os_profile, args.pkg_type)
    s3 = boto3.client("s3", endpoint_url=args.endpoint_url)
    print(f"Uploading {args.file} -> s3://{args.bucket}/{key}")
    s3.upload_file(str(args.file), args.bucket, key)
    # Confirm the object exists so a job bounded by continue-on-error cannot
    # silently succeed after a failed upload.
    s3.head_object(Bucket=args.bucket, Key=key)
    print(f"Published: s3://{args.bucket}/{key}")
    return key


def main(argv=None) -> None:
    publish(parse_args(argv))


if __name__ == "__main__":
    main()
