# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests validating workflow permissions for reusable workflow calls.

Workflows calling reusable workflows must have at least the permissions
declared by the callee. GitHub's permission model only allows permissions
to be downgraded (not elevated) in nested workflows.

This test validates that all transitive callers of reusable workflows
have the required permissions declared by those callees.

See: https://github.com/ROCm/TheRock/issues/7235
"""

import unittest

from workflow_utils import (
    WORKFLOWS_DIR,
    get_transitive_workflow_uses,
    load_workflow,
)


def get_workflow_permissions(workflow: dict) -> dict:
    """Extract the top-level permissions block from a workflow."""
    permissions = workflow.get("permissions")
    if permissions is None:
        return {}
    if isinstance(permissions, str):
        # Handle "read-all", "write-all"
        return {"_all_": permissions}
    if isinstance(permissions, dict):
        return permissions
    return {}


def permission_satisfies(caller_level: str, callee_level: str) -> bool:
    """Check if caller permission level satisfies callee requirement.

    write >= read >= none
    """
    if callee_level == "none" or callee_level == "":
        return True
    if callee_level == "read":
        return caller_level in ("read", "write")
    if callee_level == "write":
        return caller_level == "write"
    return caller_level == callee_level


def caller_satisfies_permissions(
    caller_permissions: dict, callee_permissions: dict
) -> list[str]:
    """Check if caller has all permissions required by callee.

    Returns list of error messages for missing/insufficient permissions.
    """
    errors = []

    # Handle read-all / write-all
    caller_all = caller_permissions.get("_all_", "")
    callee_all = callee_permissions.get("_all_", "")

    if callee_all in ("read-all", "write-all"):
        if caller_all not in ("read-all", "write-all"):
            errors.append(
                f"callee requires '{callee_all}' but caller has '{caller_all or 'none'}'"
            )
        return errors

    # Check each permission the callee requires
    for perm_name, callee_level in callee_permissions.items():
        if perm_name == "_all_":
            continue

        # Determine caller's level for this permission
        if caller_all == "write-all":
            caller_level = "write"
        elif caller_all == "read-all":
            caller_level = "read"
        else:
            caller_level = caller_permissions.get(perm_name, "")

        if not permission_satisfies(caller_level, callee_level):
            errors.append(
                f"'{perm_name}: {callee_level}' required but caller has "
                f"'{perm_name}: {caller_level or 'none'}'"
            )

    return errors


def find_local_workflow_calls(workflow: dict) -> list[str]:
    """Find local reusable workflow calls (uses: ./.github/workflows/...)."""
    calls = []
    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict):
        return calls

    for job_def in jobs.values():
        if not isinstance(job_def, dict):
            continue
        uses = job_def.get("uses")
        if isinstance(uses, str) and uses.startswith("./.github/workflows/"):
            filename = uses.removeprefix("./.github/workflows/")
            calls.append(filename)
    return calls


class WorkflowPermissionsTest(unittest.TestCase):
    """Verifies callers have permissions required by callees."""

    def test_callers_have_required_permissions(self):
        """All callers must have permissions declared by their callees."""
        errors = []

        for workflow_path in sorted(WORKFLOWS_DIR.glob("*.yml")):
            workflow = load_workflow(workflow_path)
            caller_permissions = get_workflow_permissions(workflow)
            called_workflows = find_local_workflow_calls(workflow)

            for callee_filename in called_workflows:
                callee_path = WORKFLOWS_DIR / callee_filename
                if not callee_path.exists():
                    continue

                callee_workflow = load_workflow(callee_path)
                callee_permissions = get_workflow_permissions(callee_workflow)

                if not callee_permissions:
                    continue

                permission_errors = caller_satisfies_permissions(
                    caller_permissions, callee_permissions
                )

                for err in permission_errors:
                    errors.append(f"{workflow_path.name} -> {callee_filename}: {err}")

        if errors:
            self.fail(
                "Workflows missing permissions required by callees:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    def test_transitive_callers_have_required_permissions(self):
        """Permissions must be satisfied through entire call chain."""
        # Find all root workflows (those with workflow_dispatch or push/pull triggers)
        root_workflows = []
        for workflow_path in sorted(WORKFLOWS_DIR.glob("*.yml")):
            workflow = load_workflow(workflow_path)
            on_block = workflow.get("on") or workflow.get(True)
            if isinstance(on_block, dict):
                # Has triggers, could be a root
                if any(
                    k in on_block
                    for k in ["push", "pull_request", "workflow_dispatch", "schedule"]
                ):
                    root_workflows.append(workflow_path.name)

        errors = []
        checked = set()

        for root in root_workflows:
            all_in_chain = get_transitive_workflow_uses([root])

            for caller_filename in all_in_chain:
                caller_path = WORKFLOWS_DIR / caller_filename
                if not caller_path.exists():
                    continue

                caller_workflow = load_workflow(caller_path)
                caller_permissions = get_workflow_permissions(caller_workflow)
                called_workflows = find_local_workflow_calls(caller_workflow)

                for callee_filename in called_workflows:
                    check_key = (caller_filename, callee_filename)
                    if check_key in checked:
                        continue
                    checked.add(check_key)

                    callee_path = WORKFLOWS_DIR / callee_filename
                    if not callee_path.exists():
                        continue

                    callee_workflow = load_workflow(callee_path)
                    callee_permissions = get_workflow_permissions(callee_workflow)

                    if not callee_permissions:
                        continue

                    permission_errors = caller_satisfies_permissions(
                        caller_permissions, callee_permissions
                    )

                    for err in permission_errors:
                        errors.append(f"{caller_filename} -> {callee_filename}: {err}")

        if errors:
            self.fail(
                "Transitive permission violations:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )


if __name__ == "__main__":
    unittest.main()
