# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Loads runner label overrides from a checked-out git repository.
Falls back gracefully if config file is not found.

This module enables dynamic runner configuration without requiring PRs to TheRock.
Configuration is stored in a separate git repository (e.g., TheRock-config) and
checked out during CI workflow execution, providing full traceability via git SHA.

Expected workflow usage:
    - name: Checkout runner config
      uses: actions/checkout@v4
      with:
        repository: ROCm/therock-runner-config
        path: .runner-config

Environment variables:
- THEROCK_RUNNER_CONFIG_PATH: Custom path to config file (for testing/overrides)
- THEROCK_DISABLE_RUNNER_OVERRIDES: Set to "1" to skip loading (for local dev/debugging)
"""

import copy
import json
import os
from pathlib import Path

from github_actions_utils import str2bool

# Default path where GitHub Actions checks out the config repo
DEFAULT_CONFIG_PATH = ".runner-config/runner-config.json"

# Module-level cache (one load per process)
_cached_overrides: dict | None = None
_load_attempted: bool = False


def _get_config_path() -> Path:
    """Get the path to runner config file, allowing override via environment variable."""
    config_path = os.environ.get("THEROCK_RUNNER_CONFIG_PATH", DEFAULT_CONFIG_PATH)
    return Path(config_path)


def _is_disabled() -> bool:
    """Check if runner overrides are disabled via environment variable."""
    return str2bool(os.environ.get("THEROCK_DISABLE_RUNNER_OVERRIDES", "false"))


def load_overrides() -> dict:
    """Load overrides from checked-out config repo. Returns empty dict on failure.

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
    global _cached_overrides, _load_attempted

    if _is_disabled():
        return {}

    if _load_attempted:
        return _cached_overrides or {}

    _load_attempted = True

    config_path = _get_config_path()

    try:
        if not config_path.exists():
            print(f"Runner config not found at {config_path}, using defaults")
            return {}

        with open(config_path) as f:
            data = json.load(f)
            _cached_overrides = data.get("overrides", {})
            print(f"Loaded runner overrides from {config_path}")
            return _cached_overrides
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: Failed to load runner overrides from {config_path}: {e}")
        return {}


def apply_overrides(family_matrix: dict) -> dict:
    """Apply config overrides to a family matrix."""
    overrides = load_overrides()

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
