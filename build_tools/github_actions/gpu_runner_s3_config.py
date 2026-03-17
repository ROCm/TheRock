# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Fetches runner label overrides from S3.
Falls back gracefully if S3 is unreachable.

This module enables dynamic runner configuration without requiring PRs to TheRock.
Overrides are stored in a public S3 bucket and fetched at runtime during CI configuration.

Environment variables:
- THEROCK_RUNNER_OVERRIDE_URL: Custom URL for override file (for testing)
- THEROCK_DISABLE_RUNNER_OVERRIDES: Set to "1" to skip fetching (for local dev/debugging)
"""

import copy
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from github_actions_utils import str2bool

# Public HTTPS URL (no auth needed for reads)
DEFAULT_OVERRIDE_URL = (
    "https://therock-ci-config.s3.amazonaws.com/therock-runner-config.json"
)

# Module-level cache (one fetch per process)
_cached_overrides: dict | None = None
_fetch_attempted: bool = False


def _get_override_url() -> str:
    """Get the URL for runner overrides, allowing override via environment variable."""
    return os.environ.get("THEROCK_RUNNER_OVERRIDE_URL", DEFAULT_OVERRIDE_URL)


def _is_disabled() -> bool:
    """Check if runner overrides are disabled via environment variable."""
    return str2bool(os.environ.get("THEROCK_DISABLE_RUNNER_OVERRIDES", "false"))


def fetch_overrides() -> dict:
    """Fetch overrides from S3. Returns empty dict on failure.

    Returns:
        Dict mapping family keys to platform overrides, e.g.:
        {
            "gfx94x": {
                "linux": {
                    "test-runs-on": "linux-mi325-1gpu-ossci-rocm",
                    ...
                }
            }
        }
    """
    global _cached_overrides, _fetch_attempted

    if _is_disabled():
        return {}

    if _fetch_attempted:
        return _cached_overrides or {}

    _fetch_attempted = True

    override_url = _get_override_url()

    try:
        req = Request(override_url, headers={"User-Agent": "TheRock-CI"})
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            _cached_overrides = data.get("overrides", {})
            print(f"Loaded runner overrides from {override_url}")
            return _cached_overrides
    except (URLError, HTTPError, json.JSONDecodeError, TimeoutError, OSError) as e:
        print(f"Warning: Failed to fetch runner overrides from {override_url}: {e}")
        return {}


def apply_overrides(family_matrix: dict) -> dict:
    """Apply S3 overrides to a family matrix."""
    overrides = fetch_overrides()

    if not overrides:
        return family_matrix

    # Deep copy to avoid mutating the original matrix
    result = copy.deepcopy(family_matrix)

    for family_key, family_overrides in overrides.items():
        if family_key not in result:
            continue

        if not isinstance(family_overrides, dict):
            continue

        for platform, platform_overrides in family_overrides.items():
            if platform not in result[family_key]:
                continue

            if not isinstance(platform_overrides, dict):
                continue

            # Merge overrides into existing config (sparse merge)
            result[family_key][platform].update(platform_overrides)

    return result


def reset_cache() -> None:
    """Reset the module-level cache. Useful for testing."""
    global _cached_overrides, _fetch_attempted
    _cached_overrides = None
    _fetch_attempted = False


def generate_overrides_json() -> str:
    """Generate runner-overrides.json content from amdgpu_family_matrix.py."""
    from amdgpu_family_matrix import (
        amdgpu_family_info_matrix_nightly,
        amdgpu_family_info_matrix_postsubmit,
        amdgpu_family_info_matrix_presubmit,
    )

    overrides = {}
    matrices = [
        amdgpu_family_info_matrix_presubmit,
        amdgpu_family_info_matrix_postsubmit,
        amdgpu_family_info_matrix_nightly,
    ]

    for matrix in matrices:
        for family_key, platforms in matrix.items():
            if family_key not in overrides:
                overrides[family_key] = {}
            for platform, config in platforms.items():
                overrides[family_key][platform] = dict(config)

    return json.dumps({"overrides": overrides}, indent=2, sort_keys=True)
