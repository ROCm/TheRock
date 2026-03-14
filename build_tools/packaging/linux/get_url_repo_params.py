#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Get URL/repo parameters: base URL from any URL, repo_sub_folder from an S3 prefix, or full repo URL from components.

Output is always KEY=value (suitable for GITHUB_OUTPUT).

Subcommands (get operations):

  get-base-url         Get base URL (scheme + netloc) from an input URL. Prints repo_base_url=<value>.
  get-repo-sub-folder  Get repo_sub_folder from an S3 prefix (last segment if YYYYMMDD-<id>, else empty). Prints repo_sub_folder=<value>.
  get-repo-url         Get full repo URL from components(release_type, native_package_type, repo_base_url, os_profile, repo_sub_folder). Prints repo_url=<value>.

Usage:
  python build_tools/packaging/linux/get_url_repo_params.py get-base-url --from-url <url>
  python build_tools/packaging/linux/get_url_repo_params.py get-repo-sub-folder --from-s3-prefix <prefix>
  python build_tools/packaging/linux/get_url_repo_params.py get-repo-url ...

Examples:
  python build_tools/packaging/linux/get_url_repo_params.py get-base-url --from-url https://example.com/v2/whl
  python build_tools/packaging/linux/get_url_repo_params.py get-repo-sub-folder --from-s3-prefix v3/packages/deb/20260204-12345
  python build_tools/packaging/linux/get_url_repo_params.py get-repo-url --release-type prerelease --native-package-type deb --repo-base-url https://x.com --os-profile ubuntu2404 --repo-sub-folder ''
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
        base_url = get_base_url(args.from_url)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"repo_base_url={base_url}")
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
    print(f"repo_sub_folder={repo_sub_folder}")
    return 0


# --- repo_url ---


def get_repo_url(
    release_type: str,
    native_package_type: str,
    repo_base_url: str,
    os_profile: str,
    repo_sub_folder: str,
) -> str:
    """
    Return the full repo URL for install tests.
    - prerelease + deb: repo_base_url / os_profile
    - prerelease + rpm: repo_base_url / os_profile / x86_64/
    - non-prerelease + deb: repo_base_url / deb / repo_sub_folder /
    - non-prerelease + rpm: repo_base_url / rpm / repo_sub_folder / x86_64/
    """
    base = repo_base_url.rstrip("/")
    if release_type == "prerelease":
        if native_package_type == "deb":
            return f"{base}/{os_profile}"
        return f"{base}/{os_profile}/x86_64/"
    if native_package_type == "deb":
        return f"{base}/deb/{repo_sub_folder}/"
    return f"{base}/rpm/{repo_sub_folder}/x86_64/"


def cmd_repo_url(args: argparse.Namespace) -> int:
    try:
        url = get_repo_url(
            release_type=args.release_type,
            native_package_type=args.native_package_type,
            repo_base_url=args.repo_base_url,
            os_profile=args.os_profile,
            repo_sub_folder=args.repo_sub_folder or "",
        )
    except (ValueError, TypeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"repo_url={url}")
    return 0


# --- main ---


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Get URL/repo parameters: base URL (from any URL) or repo_sub_folder (from S3 prefix). Output is KEY=value for GITHUB_OUTPUT.",
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Get operation to run"
    )

    # get-base-url: get base URL from any input URL
    p_base = subparsers.add_parser(
        "get-base-url",
        help="Get base URL (scheme + netloc) from an input URL; path/query/fragment are stripped.",
    )
    p_base.add_argument(
        "--from-url",
        type=str,
        required=True,
        metavar="URL",
        help="Any URL to derive base URL from (scheme + netloc only; e.g. https://example.com/v2/whl → https://example.com)",
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

    # get-repo-url: full repo URL from components (replaces inline logic in workflows)
    p_url = subparsers.add_parser(
        "get-repo-url",
        help="Get full repo URL from release_type, native_package_type, repo_base_url, os_profile, repo_sub_folder.",
    )
    p_url.add_argument(
        "--release-type", type=str, required=True, help="e.g. prerelease, dev, nightly"
    )
    p_url.add_argument(
        "--native-package-type",
        type=str,
        required=True,
        choices=["deb", "rpm"],
        help="Package type (deb or rpm)",
    )
    p_url.add_argument(
        "--repo-base-url",
        type=str,
        required=True,
        metavar="URL",
        help="Base URL (scheme + netloc, no trailing slash)",
    )
    p_url.add_argument(
        "--os-profile",
        type=str,
        required=True,
        help="OS profile (e.g. ubuntu2404, rhel9)",
    )
    p_url.add_argument(
        "--repo-sub-folder",
        type=str,
        default="",
        help="Repo subfolder (e.g. YYYYMMDD-<id> for dev/nightly; empty for prerelease)",
    )
    p_url.set_defaults(func=cmd_repo_url)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
