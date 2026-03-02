#!/usr/bin/env python
"""Sets up all parameters for the Linux runfile installer workflow.

This script handles:
1. Reading workflow inputs from environment variables
2. Computing derived values (e.g., rocm_version from version.json)
3. Auto-detecting pull_run_id and pull_tag from nightly indexes if not provided
4. Auto-detecting GFX architectures if gfx_archs is "all"

Environment variable inputs:
    * ROCM_VERSION: ROCm version (optional, reads from version.json if empty)
    * GFX_ARCHS: GFX architectures to build ("all" for auto-detect)
    * PULL_AMDGPU: Version of amdgpu to package
    * PULL_TAG: Build date in YYYYMMDD format (optional, defaults to today)
    * PULL_RUN_ID: Workflow run ID (optional, auto-detected if empty)
    * PULL_PKG: Base package name

Outputs written to GITHUB_OUTPUT:
    * rocm_version: The ROCm version
    * gfx_archs: GFX architectures (comma-separated)
    * pull_amdgpu: AMDGPU version
    * pull_pkg: Package name
    * pull_tag: Build date in YYYYMMDD format
    * pull_run_id: GitHub Actions run ID

Example usage:
    python build_tools/github_actions/setup_runfile_installer.py
"""

import json
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


def get_rocm_version_from_file() -> str:
    """Read rocm-version from version.json."""
    version_file = Path(__file__).parent.parent.parent / "version.json"
    if not version_file.exists():
        print(f"ERROR: version.json not found at {version_file}")
        sys.exit(1)

    with open(version_file) as f:
        data = json.load(f)

    version = data.get("rocm-version")
    if not version:
        print("ERROR: rocm-version not found in version.json")
        sys.exit(1)

    return version


def fetch_index(url: str, retries: int = 3) -> str:
    """Fetch HTML content from URL.
    
    Args:
        url: The URL to fetch
        retries: Number of retry attempts on failure
    """
    print(f"Fetching {url}")
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=30) as response:
                return response.read().decode("utf-8")
        except (HTTPError, URLError) as e:
            print(f"Error fetching {url} (attempt {attempt + 1}/{retries}): {e}")
            if attempt + 1 >= retries:
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
    parts = re.findall(r'(\d+|[a-zA-Z]+)', gfx)
    result = []
    for part in parts:
        if part.isdigit():
            result.append((0, int(part)))  # Numbers sort first, by value
        else:
            result.append((1, part))  # Letters sort after, alphabetically
    return result


def fetch_nightly_run_id(pull_tag: str) -> tuple[str, str]:
    """Fetch the latest nightly run ID for the given date.

    Returns:
        Tuple of (run_id, pull_tag)
    """
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

    return run_id, pull_tag


def fetch_gfx_archs(pull_tag: str, run_id: str) -> str:
    """Detect common GFX architectures from nightly package indexes.

    Returns:
        Comma-separated list of GFX architectures
    """
    print("Detecting GFX architectures...")

    folder = f"{pull_tag}-{run_id}"
    deb_pkg_url = f"{NIGHTLY_BASE_URL}/deb/{folder}/pool/main/index.html"
    rpm_pkg_url = f"{NIGHTLY_BASE_URL}/rpm/{folder}/x86_64/index.html"

    try:
        deb_pkg_html = fetch_index(deb_pkg_url)
        rpm_pkg_html = fetch_index(rpm_pkg_url)
    except (HTTPError, URLError) as e:
        print(f"ERROR: Failed to fetch package indexes: {e}")
        sys.exit(1)

    deb_gfx = extract_gfx_archs(deb_pkg_html)
    rpm_gfx = extract_gfx_archs(rpm_pkg_html)

    print(f"DEB GFX archs: {sorted(deb_gfx, key=version_sort_key)}")
    print(f"RPM GFX archs: {sorted(rpm_gfx, key=version_sort_key)}")

    common_gfx = deb_gfx & rpm_gfx
    if not common_gfx:
        print("ERROR: No common GFX architectures found")
        sys.exit(1)

    gfx_archs = ",".join(sorted(common_gfx, key=version_sort_key))
    print(f"Common GFX architectures: {gfx_archs}")

    return gfx_archs


def main():
    # Read environment variables
    rocm_version = os.environ.get("ROCM_VERSION", "")
    gfx_archs = os.environ.get("GFX_ARCHS", "all")
    pull_amdgpu = os.environ.get("PULL_AMDGPU", "release,31.10")
    pull_tag = os.environ.get("PULL_TAG", "")
    pull_run_id = os.environ.get("PULL_RUN_ID", "")
    pull_pkg = os.environ.get("PULL_PKG", "amdrocm-core-sdk")

    outputs = {}

    # Compute rocm_version if not provided
    if not rocm_version:
        rocm_version = get_rocm_version_from_file()
    print(f"Using ROCM_VERSION={rocm_version}")
    outputs["rocm_version"] = rocm_version

    # Pass through pull_amdgpu and pull_pkg
    print(f"Using PULL_AMDGPU={pull_amdgpu}")
    outputs["pull_amdgpu"] = pull_amdgpu

    print(f"Using PULL_PKG={pull_pkg}")
    outputs["pull_pkg"] = pull_pkg

    # Auto-detect pull_run_id and pull_tag if needed
    if pull_run_id and pull_tag:
        print(f"Using provided PULL_RUN_ID={pull_run_id}")
        print(f"Using provided PULL_TAG={pull_tag}")
    else:
        # Use provided pull_tag or default to today
        if not pull_tag:
            pull_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
        pull_run_id, pull_tag = fetch_nightly_run_id(pull_tag)

    outputs["pull_run_id"] = pull_run_id
    outputs["pull_tag"] = pull_tag

    # Handle gfx_archs
    if gfx_archs == "all":
        gfx_archs = fetch_gfx_archs(pull_tag, pull_run_id)
    else:
        print(f"Using provided GFX_ARCHS={gfx_archs}")

    outputs["gfx_archs"] = gfx_archs

    # Write all outputs
    gha_set_output(outputs)


if __name__ == "__main__":
    main()
