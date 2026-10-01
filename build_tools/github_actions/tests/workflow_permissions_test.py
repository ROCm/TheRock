# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests validating workflow permissions for reusable workflow calls and actions.

Workflows calling reusable workflows must have at least the permissions
declared by the callee. GitHub's permission model only allows permissions
to be downgraded (not elevated) in nested workflows.

This test validates that all transitive callers of reusable workflows
have the required permissions declared by those callees.

Additionally, actions like benc-uk/workflow-dispatch require `actions: write`
permission to trigger workflows via the GitHub API when using GITHUB_TOKEN.

See: https://github.com/ROCm/TheRock/issues/7235
See: https://github.com/ROCm/TheRock/pull/8664#issuecomment-5936450329
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


def get_job_permissions(job_def: dict, workflow_permissions: dict) -> dict:
    """Get effective permissions for a job.

    Job-level permissions override workflow-level permissions.
    If job has no permissions block, inherit from workflow.
    """
    job_permissions = job_def.get("permissions")
    if job_permissions is None:
        return workflow_permissions
    if isinstance(job_permissions, str):
        return {"_all_": job_permissions}
    if isinstance(job_permissions, dict):
        return job_permissions
    return workflow_permissions


def find_local_workflow_calls_with_permissions(
    workflow: dict,
) -> list[tuple[str, str, dict]]:
    """Find local reusable workflow calls with their effective permissions.

    Returns list of (job_name, callee_filename, effective_permissions) tuples.
    """
    calls = []
    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict):
        return calls

    workflow_permissions = get_workflow_permissions(workflow)

    for job_name, job_def in jobs.items():
        if not isinstance(job_def, dict):
            continue
        uses = job_def.get("uses")
        if isinstance(uses, str) and uses.startswith("./.github/workflows/"):
            filename = uses.removeprefix("./.github/workflows/")
            effective_permissions = get_job_permissions(job_def, workflow_permissions)
            calls.append((job_name, filename, effective_permissions))
    return calls


class WorkflowPermissionsTest(unittest.TestCase):
    """Verifies callers have permissions required by callees."""

    def test_callers_have_required_permissions(self):
        """All callers must have permissions declared by their callees."""
        errors = []

        for workflow_path in sorted(WORKFLOWS_DIR.glob("*.yml")):
            workflow = load_workflow(workflow_path)
            calls = find_local_workflow_calls_with_permissions(workflow)

            for job_name, callee_filename, caller_permissions in calls:
                callee_path = WORKFLOWS_DIR / callee_filename
                # Skip non-local workflows (e.g., cross-repo references won't exist locally)
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
                    errors.append(
                        f"{workflow_path.name}:{job_name} -> {callee_filename}: {err}"
                    )

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
                # Skip non-local workflows (e.g., cross-repo references won't exist locally)
                if not caller_path.exists():
                    continue

                caller_workflow = load_workflow(caller_path)
                calls = find_local_workflow_calls_with_permissions(caller_workflow)

                for job_name, callee_filename, caller_permissions in calls:
                    check_key = (caller_filename, job_name, callee_filename)
                    if check_key in checked:
                        continue
                    checked.add(check_key)

                    callee_path = WORKFLOWS_DIR / callee_filename
                    # Skip non-local workflows (e.g., cross-repo references won't exist locally)
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
                        errors.append(
                            f"{caller_filename}:{job_name} -> {callee_filename}: {err}"
                        )

        if errors:
            self.fail(
                "Transitive permission violations:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )


WORKFLOW_DISPATCH_ACTION_NAME = "benc-uk/workflow-dispatch"

# Patterns that indicate use of GITHUB_TOKEN (either default or explicit)
GITHUB_TOKEN_PATTERNS = (
    "${{ github.token }}",
    "${{ secrets.GITHUB_TOKEN }}",
)


def uses_github_token(step: dict) -> bool:
    """Check if a benc-uk/workflow-dispatch step uses GITHUB_TOKEN.

    The action uses GITHUB_TOKEN by default. If a custom token is provided
    via the 'token' parameter that doesn't match GITHUB_TOKEN patterns,
    we can't validate permissions (those come from the app/PAT, not the
    workflow's permission grants).

    Returns True if the step uses GITHUB_TOKEN (default or explicit).
    """
    with_block = step.get("with", {})
    token_value = with_block.get("token")

    if token_value is None:
        # No token specified = uses default GITHUB_TOKEN
        return True

    # Check if it's an explicit GITHUB_TOKEN reference
    token_str = str(token_value).strip()
    for pattern in GITHUB_TOKEN_PATTERNS:
        if pattern in token_str:
            return True

    # Custom token (app token, PAT, etc.) - can't validate via workflow permissions
    return False


def find_dispatch_steps_with_permissions(
    workflow: dict,
) -> list[tuple[str, str, dict]]:
    """Find benc-uk/workflow-dispatch steps with their effective permissions.

    Returns list of (job_name, step_name, effective_permissions) tuples
    for steps that use GITHUB_TOKEN.
    """
    results = []
    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict):
        return results

    workflow_permissions = get_workflow_permissions(workflow)

    for job_name, job_def in jobs.items():
        if not isinstance(job_def, dict):
            continue

        job_permissions = get_job_permissions(job_def, workflow_permissions)

        for step in job_def.get("steps", []):
            if not isinstance(step, dict):
                continue
            uses = step.get("uses", "")
            if WORKFLOW_DISPATCH_ACTION_NAME not in uses:
                continue
            if not uses_github_token(step):
                continue

            step_name = step.get("name", "(unnamed)")
            results.append((job_name, step_name, job_permissions))

    return results


def has_actions_write(permissions: dict) -> bool:
    """Check if permissions include actions: write."""
    # Handle write-all
    if permissions.get("_all_") == "write-all":
        return True

    actions_level = permissions.get("actions", "")
    return actions_level == "write"


class WorkflowDispatchPermissionsTest(unittest.TestCase):
    """Verifies benc-uk/workflow-dispatch steps have actions: write permission."""

    def test_dispatch_steps_have_actions_write(self):
        """Jobs using benc-uk/workflow-dispatch with GITHUB_TOKEN need actions: write."""
        errors = []

        for workflow_path in sorted(WORKFLOWS_DIR.glob("*.yml")):
            workflow = load_workflow(workflow_path)
            dispatch_steps = find_dispatch_steps_with_permissions(workflow)

            for job_name, step_name, permissions in dispatch_steps:
                if not has_actions_write(permissions):
                    errors.append(
                        f"{workflow_path.name} / {job_name}: "
                        f"step '{step_name}' uses {WORKFLOW_DISPATCH_ACTION_NAME} "
                        f"with GITHUB_TOKEN but lacks 'actions: write' permission"
                    )

        if errors:
            self.fail(
                "Jobs using benc-uk/workflow-dispatch need 'actions: write' permission:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )


class DispatchPermissionsHelpersTest(unittest.TestCase):
    """Unit tests for dispatch permission helper functions."""

    def test_job_permission_resolution(self):
        """Test get_job_permissions resolves permissions correctly."""
        workflow_perms = {"actions": "write", "contents": "read"}

        # Job permissions override workflow entirely
        result = get_job_permissions(
            {"permissions": {"contents": "write"}}, workflow_perms
        )
        self.assertEqual(result, {"contents": "write"})
        self.assertNotIn("actions", result)

        # No job permissions = inherit from workflow
        result = get_job_permissions({"runs-on": "ubuntu-latest"}, workflow_perms)
        self.assertEqual(result, workflow_perms)

        # Empty permissions block = empty (not inherited)
        result = get_job_permissions({"permissions": {}}, workflow_perms)
        self.assertEqual(result, {})

        # String permission (write-all)
        result = get_job_permissions({"permissions": "write-all"}, {})
        self.assertEqual(result, {"_all_": "write-all"})

    def test_has_actions_write(self):
        """Test has_actions_write detects actions: write correctly."""
        # Should return True
        self.assertTrue(has_actions_write({"actions": "write"}))
        self.assertTrue(has_actions_write({"_all_": "write-all"}))

        # Should return False
        self.assertFalse(has_actions_write({"actions": "read"}))
        self.assertFalse(has_actions_write({"_all_": "read-all"}))
        self.assertFalse(has_actions_write({"contents": "write"}))
        self.assertFalse(has_actions_write({}))

    def test_uses_github_token(self):
        """Test uses_github_token detects GITHUB_TOKEN vs custom tokens."""
        # Should return True (uses GITHUB_TOKEN)
        self.assertTrue(uses_github_token({"uses": "benc-uk/workflow-dispatch@v1"}))
        self.assertTrue(uses_github_token({"with": {"workflow": "test.yml"}}))
        self.assertTrue(uses_github_token({"with": {"token": "${{ github.token }}"}}))
        self.assertTrue(
            uses_github_token({"with": {"token": "${{ secrets.GITHUB_TOKEN }}"}})
        )

        # Should return False (custom token)
        self.assertFalse(
            uses_github_token({"with": {"token": "${{ secrets.MY_PAT }}"}})
        )
        self.assertFalse(
            uses_github_token({"with": {"token": "${{ steps.app.outputs.token }}"}})
        )

    def test_find_dispatch_steps_with_permissions(self):
        """Test find_dispatch_steps_with_permissions finds steps correctly."""
        # Job permissions override workflow permissions
        workflow = {
            "permissions": {"actions": "write"},
            "jobs": {
                "trigger": {
                    "permissions": {"contents": "write"},
                    "steps": [
                        {"name": "Dispatch", "uses": "benc-uk/workflow-dispatch@v1"}
                    ],
                }
            },
        }
        result = find_dispatch_steps_with_permissions(workflow)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ("trigger", "Dispatch", {"contents": "write"}))

        # Inherits workflow permissions when job has none
        workflow["jobs"]["trigger"].pop("permissions")
        result = find_dispatch_steps_with_permissions(workflow)
        self.assertEqual(result[0][2], {"actions": "write"})

        # Skips steps with custom tokens
        workflow = {
            "jobs": {
                "trigger": {
                    "steps": [
                        {
                            "uses": "benc-uk/workflow-dispatch@v1",
                            "with": {"token": "${{ secrets.PAT }}"},
                        }
                    ]
                }
            }
        }
        self.assertEqual(find_dispatch_steps_with_permissions(workflow), [])

        # Unnamed steps get default name
        workflow = {
            "jobs": {"trigger": {"steps": [{"uses": "benc-uk/workflow-dispatch@v1"}]}}
        }
        result = find_dispatch_steps_with_permissions(workflow)
        self.assertEqual(result[0][1], "(unnamed)")


if __name__ == "__main__":
    unittest.main()
