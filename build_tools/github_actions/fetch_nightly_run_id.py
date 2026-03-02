#!/usr/bin/env python
"""Fetches the latest nightly run ID from rocm.nightlies.amd.com.

This script fetches the rpm and deb index pages and finds the latest
folder (YYYYMMDD-RUNID) that exists in both.

Environment variable inputs:
    * PULL_TAG: Date in YYYYMMDD format (optional, defaults to today)

Outputs written to GITHUB_OUTPUT:
    * pull_run_id: The GitHub Actions run ID
    * pull_tag: The date tag used
    * gfx_archs: Comma-separated list of common GFX architectures (if requested)

Example usage:
    python build_tools/github_actions/fetch_nightly_run_id.py
    python build_tools/github_actions/fetch_nightly_run_id.py --pull-tag 20260227
    python build_tools/github_actions/fetch_nightly_run_id.py --include-gfx-archs
"""

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from github_actions_utils import gha_set_output


NIGHTLY_BASE_URL = "https://rocm.nightlies.amd.com"


def fetch_index(url: str) -> str:
    """Fetch HTML content from URL."""
    print(f"Fetching {url}")
    try:
        with urlopen(url, timeout=30) as response:
            return response.read().decode("utf-8")
    except (HTTPError, URLError) as e:
        print(f"Error fetching {url}: {e}")
        raise


def extract_folders(html: str, date_prefix: str) -> set[str]:
    """Extract folder names matching YYYYMMDD-RUNID pattern."""
    pattern = rf"({date_prefix}-\d+)"
    matches = re.findall(pattern, html)
    return set(matches)


def find_common_latest(rpm_folders: set[str], deb_folders: set[str]) -> str | None:
    """Find the latest folder that exists in both sets."""
    common = rpm_folders & deb_folders
    if not common:
        return None
    # Sort by run ID (numeric) and return highest
    return max(common, key=lambda x: int(x.split("-")[1]))


def extract_gfx_archs(html: str, package_prefix: str = "amdrocm-core-sdk-") -> set[str]:
    """Extract GFX architecture names from package index HTML."""
    pattern = rf'{package_prefix}(gfx[a-z0-9]+)[_-]'
    matches = re.findall(pattern, html, re.IGNORECASE)
    return set(matches)


def version_sort_key(gfx: str):
    """Sort key for GFX architecture names with proper version ordering.
    
    Handles names like gfx90a, gfx942, gfx1100, gfx1150 correctly.
    Extracts numeric parts and sorts numerically.
    """
    # Extract all numeric and alphabetic parts
    parts = re.findall(r'(\d+|[a-zA-Z]+)', gfx)
    result = []
    for part in parts:
        if part.isdigit():
            result.append((0, int(part)))  # Numbers sort first, by value
        else:
            result.append((1, part))  # Letters sort after, alphabetically
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pull-tag", help="Date in YYYYMMDD format")
    parser.add_argument("--include-gfx-archs", action="store_true",
                        help="Also detect common GFX architectures")
    args = parser.parse_args()

    pull_tag = args.pull_tag or datetime.now(timezone.utc).strftime("%Y%m%d")

    print(f"Fetching nightly indexes for {pull_tag}...")

    rpm_html = fetch_index(f"{NIGHTLY_BASE_URL}/rpm/")
    deb_html = fetch_index(f"{NIGHTLY_BASE_URL}/deb/")

    rpm_folders = extract_folders(rpm_html, pull_tag)
    deb_folders = extract_folders(deb_html, pull_tag)

    print(f"RPM folders: {sorted(rpm_folders)}")
    print(f"DEB folders: {sorted(deb_folders)}")

    latest = find_common_latest(rpm_folders, deb_folders)
    if not latest:
        print(f"ERROR: No matching folders for {pull_tag} in both rpm and deb")
        sys.exit(1)

    run_id = latest.split("-")[1]
    print(f"Found latest: {latest} (run_id={run_id})")

    outputs = {
        "pull_run_id": run_id,
        "pull_tag": pull_tag,
    }

    # Optionally detect GFX archs
    if args.include_gfx_archs:
        print("Detecting GFX architectures...")

        deb_pkg_url = f"{NIGHTLY_BASE_URL}/deb/{latest}/pool/main/index.html"
        rpm_pkg_url = f"{NIGHTLY_BASE_URL}/rpm/{latest}/x86_64/index.html"

        try:
            deb_pkg_html = fetch_index(deb_pkg_url)
            rpm_pkg_html = fetch_index(rpm_pkg_url)
        except (HTTPError, URLError) as e:
            print(f"ERROR: Failed to fetch package indexes: {e}")
            sys.exit(1)

        deb_gfx = extract_gfx_archs(deb_pkg_html)
        rpm_gfx = extract_gfx_archs(rpm_pkg_html)

        print(f"DEB GFX archs: {sorted(deb_gfx, reverse=True)}")
        print(f"RPM GFX archs: {sorted(rpm_gfx, reverse=True)}")

        common_gfx = deb_gfx & rpm_gfx
        if not common_gfx:
            print("ERROR: No common GFX architectures found")
            sys.exit(1)

        gfx_archs = ",".join(sorted(common_gfx, key=version_sort_key))
        print(f"Common GFX architectures: {gfx_archs}")
        outputs["gfx_archs"] = gfx_archs

    gha_set_output(outputs)


if __name__ == "__main__":
    main()
