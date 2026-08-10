#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate that workflows calling setup_multi_arch.yml have actions:read.

The stage reuse feature in setup_multi_arch.yml requires `actions: read`
permission to query baseline workflow runs. Workflows that call
setup_multi_arch.yml must have this permission in their permissions block,
otherwise the stage reuse feature will fail with a 403 error.

This test parses workflow YAML files and validates the permissions are correct.

Usage:
    python test_workflow_permissions.py

See: https://github.com/ROCm/TheRock/issues/7235
"""

import sys
from pathlib import Path

import yaml

WORKFLOWS_DIR = Path(__file__).resolve().parent.parent

# Reusable workflows that require actions permission for stage reuse
WORKFLOWS_REQUIRING_ACTIONS_PERMISSION = {
    "setup_multi_arch.yml",
}


def load_workflow(path: Path) -> dict:
    """Load a YAML workflow file."""
    with open(path) as f:
        return yaml.safe_load(f)


def get_workflow_permissions(workflow: dict) -> dict:
    """Extract the top-level permissions block from a workflow."""
    permissions = workflow.get("permissions")
    if permissions is None:
        return {}
    if isinstance(permissions, str):
        return {"_all_": permissions}
    if isinstance(permissions, dict):
        return permissions
    return {}


def has_actions_read_permission(workflow: dict) -> bool:
    """Check if a workflow has actions: read (or write) permission."""
    permissions = get_workflow_permissions(workflow)

    if not permissions:
        return False

    # Check for global read/write all
    all_access = permissions.get("_all_", "")
    if all_access in ("read-all", "write-all"):
        return True

    # Check for specific actions permission
    actions_perm = permissions.get("actions", "")
    return actions_perm in ("read", "write")


def find_reusable_workflow_calls(workflow: dict) -> list[str]:
    """Find all reusable workflow calls in a workflow."""
    calls = []
    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict):
        return calls

    for job_def in jobs.values():
        if not isinstance(job_def, dict):
            continue
        uses = job_def.get("uses")
        if isinstance(uses, str) and ".github/workflows/" in uses:
            # Extract just the filename
            parts = uses.split(".github/workflows/")[-1]
            filename = parts.split("@")[0]
            calls.append(filename)
    return calls


def validate_workflow(workflow_path: Path) -> list[str]:
    """Validate a workflow has required permissions. Returns list of errors."""
    errors = []
    workflow = load_workflow(workflow_path)
    called_workflows = find_reusable_workflow_calls(workflow)

    requires_actions = any(
        called in WORKFLOWS_REQUIRING_ACTIONS_PERMISSION for called in called_workflows
    )

    if requires_actions and not has_actions_read_permission(workflow):
        errors.append(
            f"{workflow_path.name}: calls setup_multi_arch.yml but missing "
            f"'actions: read' permission. Current permissions: "
            f"{get_workflow_permissions(workflow)}"
        )

    return errors


def main() -> int:
    """Validate all workflows have required permissions."""
    print("Validating workflow permissions...")
    print(f"Scanning: {WORKFLOWS_DIR}")
    print()

    all_errors = []

    for workflow_path in sorted(WORKFLOWS_DIR.glob("*.yml")):
        errors = validate_workflow(workflow_path)
        if errors:
            all_errors.extend(errors)
            for error in errors:
                print(f"FAIL: {error}")
        else:
            called = find_reusable_workflow_calls(load_workflow(workflow_path))
            if any(c in WORKFLOWS_REQUIRING_ACTIONS_PERMISSION for c in called):
                print(f"PASS: {workflow_path.name}")

    print()
    if all_errors:
        print(f"FAILED: {len(all_errors)} workflow(s) missing required permissions")
        return 1
    else:
        print("All workflows have correct permissions!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
