#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Test that GITHUB_TOKEN has required permissions for stage reuse.

The stage reuse feature in setup_multi_arch.yml requires `actions: read`
permission to query baseline workflow runs via the GitHub Actions API.
This script validates that the permission is correctly configured.

Usage:
    python test_workflow_permissions.py

Environment:
    GITHUB_TOKEN: Required. The token to test.
    GITHUB_REPOSITORY: Optional. Defaults to ROCm/TheRock.

See: https://github.com/ROCm/TheRock/issues/7235
"""

import json
import os
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def test_actions_read_permission() -> bool:
    """Query workflow runs to verify actions:read permission works."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: GITHUB_TOKEN not set")
        return False

    repo = os.environ.get("GITHUB_REPOSITORY", "ROCm/TheRock")
    url = f"https://api.github.com/repos/{repo}/actions/runs?per_page=1"

    request = Request(url)
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("X-GitHub-Api-Version", "2022-11-28")

    try:
        with urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode())
            total_count = data.get("total_count", 0)
            print(f"SUCCESS: Retrieved workflow runs (total_count={total_count})")
            print("The actions:read permission is working correctly.")
            return True
    except HTTPError as e:
        if e.code == 403:
            print("ERROR: 403 Forbidden - Missing 'actions: read' permission")
            print("The GITHUB_TOKEN does not have permission to query workflow runs.")
            print("Ensure the workflow has 'actions: read' in its permissions block.")
        else:
            print(f"ERROR: HTTP {e.code} - {e.reason}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_workflow_specific_runs() -> bool:
    """Query runs for a specific workflow (mirrors baseline_runs.py pattern)."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: GITHUB_TOKEN not set")
        return False

    repo = os.environ.get("GITHUB_REPOSITORY", "ROCm/TheRock")
    workflow = "multi_arch_ci.yml"

    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/runs?per_page=1&status=completed"

    request = Request(url)
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("X-GitHub-Api-Version", "2022-11-28")

    try:
        with urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode())
            runs = data.get("workflow_runs", [])
            print(
                f"SUCCESS: Queried {workflow} runs (found {len(runs)} completed runs)"
            )
            print("Stage reuse API pattern is working correctly.")
            return True
    except HTTPError as e:
        if e.code == 403:
            print(f"ERROR: 403 Forbidden when querying {workflow}")
            print("Missing 'actions: read' permission for workflow-specific queries.")
            return False
        elif e.code == 404:
            # Workflow might not exist yet in a fork - that's OK
            print(f"WARNING: {workflow} not found (expected in forks)")
            return True
        else:
            print(f"ERROR: HTTP {e.code} - {e.reason}")
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main() -> int:
    """Run all permission tests."""
    print("Testing GitHub Actions API permissions...")
    print()

    success = True

    print("=== Test 1: General workflow runs query ===")
    if not test_actions_read_permission():
        success = False
    print()

    print("=== Test 2: Workflow-specific runs query (stage reuse pattern) ===")
    if not test_workflow_specific_runs():
        success = False
    print()

    if success:
        print("All permission tests passed!")
        return 0
    else:
        print("Some permission tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
