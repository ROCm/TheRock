#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Get URL/repo parameters: base URL from a CloudFront URL, or repo_sub_folder from an S3 prefix.

Subcommands (get operations):

  get-base-url         Get base URL (scheme + netloc) from a CloudFront URL. With --format env outputs repo_base_url=<value>.
  get-repo-sub-folder  Get repo_sub_folder from an S3 prefix (last segment if YYYYMMDD-<id>, else empty).

Usage:
  python build_tools/packaging/linux/get_url_repo_params.py get-base-url --from-cloudfront-url <url> [--format env|value]
  python build_tools/packaging/linux/get_url_repo_params.py get-repo-sub-folder --from-s3-prefix <prefix> [--format env|value]

Examples:
  python build_tools/packaging/linux/get_url_repo_params.py get-base-url --from-cloudfront-url https://example.com/v2/whl
  python build_tools/packaging/linux/get_url_repo_params.py get-repo-sub-folder --from-s3-prefix v3/packages/deb/20260204-12345
"""

import argparse
import re
import sys
from urllib.parse import urlparse


# --- base_url ---

def get_base_url(url: str) -> str:
    """Return base URL (scheme + netloc only). No path, query, or fragment."""
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL: {url!r}")
    return f"{parsed.scheme}://{parsed.netloc}"


def cmd_base_url(args: argparse.Namespace) -> int:
    try:
        base_url = get_base_url(args.from_cloudfront_url)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if args.format == "env":
        print(f"repo_base_url={base_url}")
    else:
        print(base_url)
    return 0


# --- repo_sub_folder ---

DATE_ARTIFACT_PATTERN = re.compile(r"^\d{8}-\d+$")


def get_repo_sub_folder(s3_prefix: str) -> str:
    """Return last path segment if it matches YYYYMMDD-<id>, else empty."""
    segments = [p for p in s3_prefix.strip("/").split("/") if p]
    if not segments:
        return ""
    last = segments[-1]
    if DATE_ARTIFACT_PATTERN.fullmatch(last):
        return last
    return ""


def cmd_repo_sub_folder(args: argparse.Namespace) -> int:
    repo_sub_folder = get_repo_sub_folder(args.from_s3_prefix)
    if args.format == "env":
        print(f"repo_sub_folder={repo_sub_folder}")
    else:
        print(repo_sub_folder)
    return 0


# --- main ---

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Get URL/repo parameters: base URL (from CloudFront URL) or repo_sub_folder (from S3 prefix).",
    )
    parser.add_argument(
        "--format",
        choices=["env", "value"],
        default="env",
        help="Output: env = KEY=value (default); value = value only",
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Get operation to run"
    )

    # get-base-url: get base URL from a CloudFront URL
    p_base = subparsers.add_parser(
        "get-base-url",
        help="Get base URL (scheme + netloc) from a CloudFront URL; path/query/fragment are stripped.",
    )
    p_base.add_argument(
        "--from-cloudfront-url",
        type=str,
        required=True,
        metavar="URL",
        help="CloudFront (or any) URL to derive base URL from (e.g. https://example.com/v2/whl → https://example.com)",
    )
    p_base.set_defaults(func=cmd_base_url)

    # get-repo-sub-folder: get repo_sub_folder from S3 prefix
    p_repo = subparsers.add_parser(
        "get-repo-sub-folder",
        help="Get repo_sub_folder from an S3 prefix (last path segment if YYYYMMDD-<id>, else empty).",
    )
    p_repo.add_argument(
        "--from-s3-prefix",
        type=str,
        required=True,
        metavar="PREFIX",
        help="S3 key prefix to derive repo_sub_folder from (e.g. v3/packages/deb/20260204-12345 → 20260204-12345)",
    )
    p_repo.set_defaults(func=cmd_repo_sub_folder)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
