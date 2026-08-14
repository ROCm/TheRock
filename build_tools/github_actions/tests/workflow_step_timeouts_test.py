# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests that checkout and fetch_sources steps have timeout-minutes set.

Long-running git operations (actions/checkout, fetch_sources.py) can hang
indefinitely if GitHub throttles or a network issue occurs. Every such step
must declare a timeout-minutes so hung jobs fail fast instead of consuming
runner capacity until the job-level timeout fires (which may be hours away).

See: ROCm/TheRock#7343
"""

import unittest

from workflow_utils import WORKFLOWS_DIR, load_workflow

# Step uses: prefixes that identify git checkout actions.
_CHECKOUT_ACTION_PREFIX = "actions/checkout@"

# Substrings in a step's run: command that identify a fetch_sources invocation.
_FETCH_SOURCES_PATTERNS = ("fetch_sources.py",)


def _step_is_checkout(step: dict) -> bool:
    """Returns True if the step uses actions/checkout."""
    uses = step.get("uses", "")
    return isinstance(uses, str) and uses.startswith(_CHECKOUT_ACTION_PREFIX)


def _step_is_fetch_sources(step: dict) -> bool:
    """Returns True if the step runs fetch_sources.py."""
    run = step.get("run", "")
    if not isinstance(run, str):
        return False
    return any(pattern in run for pattern in _FETCH_SOURCES_PATTERNS)


def _find_violations(workflow: dict, workflow_name: str) -> list[str]:
    """Returns error strings for every checkout/fetch_sources step missing a timeout."""
    errors = []
    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict):
        return errors

    for job_name, job_def in jobs.items():
        if not isinstance(job_def, dict):
            continue
        steps = job_def.get("steps")
        if not isinstance(steps, list):
            continue

        for step in steps:
            if not isinstance(step, dict):
                continue
            if not (_step_is_checkout(step) or _step_is_fetch_sources(step)):
                continue
            if step.get("timeout-minutes") is None:
                step_name = step.get("name") or step.get("uses") or step.get("run", "")[:60]
                errors.append(
                    f"{workflow_name} / {job_name} / '{step_name}': "
                    f"missing timeout-minutes"
                )

    return errors


class WorkflowStepTimeoutsTest(unittest.TestCase):
    """Verifies that checkout and fetch_sources steps declare timeout-minutes."""

    def test_checkout_and_fetch_sources_have_timeouts(self):
        """Every actions/checkout and fetch_sources.py step must have timeout-minutes."""
        errors = []

        for workflow_path in sorted(WORKFLOWS_DIR.glob("*.yml")):
            workflow = load_workflow(workflow_path)
            errors.extend(_find_violations(workflow, workflow_path.name))

        if errors:
            self.fail(
                "The following checkout/fetch_sources steps are missing timeout-minutes:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )


if __name__ == "__main__":
    unittest.main()
