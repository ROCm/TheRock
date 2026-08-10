# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests validating workflow permissions for reusable workflow calls.

Workflows that call setup_multi_arch.yml require `actions: read` permission
for the stage reuse feature, which queries the GitHub Actions API to find
baseline workflow runs. This requirement was missed during initial development
and caught in production (see issue #7235).

This test validates that:
1. Workflows calling setup_multi_arch.yml have `actions: read` (or `actions: write`)
2. Future additions are caught before they break in production

The test parses workflow YAML files and checks the top-level permissions block.
"""

from pathlib import Path
import unittest

from workflow_utils import WORKFLOWS_DIR, load_workflow

# Reusable workflows that require actions permission for stage reuse
WORKFLOWS_REQUIRING_ACTIONS_PERMISSION = {
    "setup_multi_arch.yml",
}


def get_workflow_permissions(workflow: dict) -> dict:
    """Extract the top-level permissions block from a workflow.

    Returns:
        A dict of permission_name -> access_level, or empty dict if no permissions block.
        When permissions is a string (like "read-all"), returns {"_all_": value}.
    """
    permissions = workflow.get("permissions")
    if permissions is None:
        return {}
    if isinstance(permissions, str):
        # Handle "read-all", "write-all", etc.
        return {"_all_": permissions}
    if isinstance(permissions, dict):
        return permissions
    return {}


def has_actions_read_permission(workflow: dict) -> bool:
    """Check if a workflow has actions: read (or write) permission.

    Returns True if:
    - permissions.actions is "read" or "write"
    - permissions is "read-all" or "write-all"
    - No permissions block at all (defaults to full permissions in some contexts)
    """
    permissions = get_workflow_permissions(workflow)

    # No explicit permissions = depends on context, may have full permissions
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
    """Find all reusable workflow calls (uses: ./.github/workflows/...) in a workflow.

    Returns:
        List of workflow filenames being called (e.g., ["setup_multi_arch.yml"])
    """
    calls = []
    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict):
        return calls

    for job_def in jobs.values():
        if not isinstance(job_def, dict):
            continue
        uses = job_def.get("uses")
        if isinstance(uses, str):
            # Handle both local (./.github/workflows/) and remote (owner/repo/.github/workflows/) refs
            # Local: ./.github/workflows/setup_multi_arch.yml
            # Remote: ROCm/TheRock/.github/workflows/setup_multi_arch.yml@main
            if ".github/workflows/" in uses:
                # Extract just the filename
                parts = uses.split(".github/workflows/")[-1]
                # Remove @ref suffix if present
                filename = parts.split("@")[0]
                calls.append(filename)
    return calls


class WorkflowPermissionsTest(unittest.TestCase):
    """Tests that workflows have required permissions for their reusable workflow calls."""

    pass


def _make_actions_permission_test(workflow_path: Path):
    """Create a test that verifies a workflow has actions permission when needed."""

    def test_method(self):
        workflow = load_workflow(workflow_path)
        called_workflows = find_reusable_workflow_calls(workflow)

        # Check if this workflow calls any that require actions permission
        requires_actions = any(
            called in WORKFLOWS_REQUIRING_ACTIONS_PERMISSION
            for called in called_workflows
        )

        if not requires_actions:
            # This workflow doesn't call any workflows requiring actions permission
            return

        # This workflow DOES call setup_multi_arch.yml (or similar), so it needs actions: read
        if not has_actions_read_permission(workflow):
            self.fail(
                f"{workflow_path.name} calls a workflow requiring 'actions: read' permission "
                f"(one of: {sorted(WORKFLOWS_REQUIRING_ACTIONS_PERMISSION)}), "
                f"but does not have 'actions: read' or 'actions: write' in its permissions block. "
                f"The stage reuse feature requires this permission to query workflow runs via the GitHub API. "
                f"Current permissions: {get_workflow_permissions(workflow)}"
            )

    return test_method


def _workflow_name_to_test_suffix(workflow_path: Path) -> str:
    """Converts a workflow filename to a valid Python identifier suffix."""
    return workflow_path.stem.replace("-", "_").replace(".", "_")


# Dynamically generate test methods for all workflow files
for _workflow_path in sorted(WORKFLOWS_DIR.glob("*.yml")):
    _suffix = _workflow_name_to_test_suffix(_workflow_path)
    _test = _make_actions_permission_test(_workflow_path)
    _test.__doc__ = f"Verify {_workflow_path.name} has required actions permission"
    setattr(
        WorkflowPermissionsTest,
        f"test_actions_permission__{_suffix}",
        _test,
    )


if __name__ == "__main__":
    unittest.main()
