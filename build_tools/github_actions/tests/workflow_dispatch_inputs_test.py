# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests validating inputs to benc-uk/workflow-dispatch in workflow files.

The https://github.com/benc-uk/workflow-dispatch action triggers workflows via
the GitHub REST API's "Create a workflow dispatch event" endpoint. This endpoint
rejects any inputs that are not defined in the target workflow's
`on: workflow_dispatch: inputs:` section. Changes to workflow files that
pass basic parsing checks and https://github.com/rhysd/actionlint can still
fail at runtime, like the bug at
https://github.com/ROCm/TheRock/pull/2557#discussion_r2717677336. These tests
add an extra layer of validation.

This file creates test cases for each file in .github/workflows/ that uses
benc-uk/workflow-dispatch. It is run like a standard unit test.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import unittest

from workflow_utils import (
    WORKFLOWS_DIR,
    get_choice_options,
    get_required_workflow_dispatch_inputs,
    get_workflow_dispatch_inputs,
    load_workflow,
)

WORKFLOW_DISPATCH_ACTION_NAME = "benc-uk/workflow-dispatch"


def parse_dispatch_inputs_json(inputs_raw: str) -> dict:
    """Parses the JSON inputs string from a benc-uk/workflow-dispatch step.

    For an action step with:
        uses: benc-uk/workflow-dispatch@v1.2.4
        with:
          inputs: |
            { "amdgpu_family": "${{ matrix.amdgpu_family }}",
              "release_type": "dev" }

    Returns: {"amdgpu_family": "${{ matrix.amdgpu_family }}", "release_type": "dev"}
    """
    if not inputs_raw:
        return {}

    parsed = json.loads(inputs_raw)
    if isinstance(parsed, dict):
        return parsed

    return {}


@dataclass
class DispatchCall:
    """A single benc-uk/workflow-dispatch action invocation."""

    step_name: str
    target_workflow: str
    passed_inputs: dict


def find_dispatch_calls_in_workflow(workflow: dict) -> list[DispatchCall]:
    """Finds benc-uk/workflow-dispatch steps in a single workflow."""
    if not workflow or "jobs" not in workflow:
        return []

    calls = []
    for job_name, job in workflow["jobs"].items():
        for step in job.get("steps", []):
            uses = step.get("uses", "")
            if WORKFLOW_DISPATCH_ACTION_NAME not in uses:
                continue

            with_block = step.get("with", {})
            calls.append(
                DispatchCall(
                    step_name=step.get("name", "(unnamed)"),
                    target_workflow=with_block.get("workflow", ""),
                    passed_inputs=parse_dispatch_inputs_json(
                        with_block.get("inputs", "")
                    ),
                )
            )
    return calls


class WorkflowDispatchInputsTest(unittest.TestCase):
    """Verifies benc-uk/workflow-dispatch calls only pass valid inputs.

    Test cases are generated dynamically, one per workflow file.
    """

    pass


def _make_unexpected_inputs_test(workflow_path: Path):
    """Creates a test that checks for unexpected inputs in dispatch calls."""

    def test_method(self):
        workflow = load_workflow(workflow_path)
        calls = find_dispatch_calls_in_workflow(workflow)
        errors = []
        for call in calls:
            # benc-uk/workflow-dispatch supports workflow names, filenames, or
            # IDs. We enforce filenames so we can resolve and validate the
            # target workflow locally.
            target_path = WORKFLOWS_DIR / call.target_workflow
            if not target_path.exists():
                errors.append(
                    f"step '{call.step_name}' dispatches "
                    f"'{call.target_workflow}' which does not exist"
                )
                continue

            target_workflow = load_workflow(target_path)
            accepted_inputs = get_workflow_dispatch_inputs(target_workflow)
            unexpected = call.passed_inputs.keys() - accepted_inputs
            if unexpected:
                errors.append(
                    f"step '{call.step_name}' passes unexpected inputs to "
                    f"'{call.target_workflow}': {sorted(unexpected)}. "
                    f"Accepted: {sorted(accepted_inputs)}"
                )

        if errors:
            self.fail("\n".join(errors))

    return test_method


def _make_required_inputs_test(workflow_path: Path):
    """Creates a test that checks all required inputs are passed."""

    def test_method(self):
        workflow = load_workflow(workflow_path)
        calls = find_dispatch_calls_in_workflow(workflow)
        errors = []
        for call in calls:
            target_path = WORKFLOWS_DIR / call.target_workflow
            if not target_path.exists():
                continue

            target_workflow = load_workflow(target_path)
            required_inputs = get_required_workflow_dispatch_inputs(target_workflow)
            missing = required_inputs - call.passed_inputs.keys()
            if missing:
                errors.append(
                    f"step '{call.step_name}' does not pass required inputs to "
                    f"'{call.target_workflow}': {sorted(missing)}"
                )

        if errors:
            self.fail("\n".join(errors))

    return test_method


def _is_expression(value: str) -> bool:
    """Returns True if the value contains a GitHub Actions expression."""
    return "${{" in str(value)


def _make_literal_choice_values_test(workflow_path: Path):
    """Creates a test that checks literal values are valid for choice inputs.

    When a dispatch step passes a literal string (not a ${{ }} expression) to a
    target input that is type: choice, the literal must be in the target's
    allowed options list. GitHub rejects invalid choice values at dispatch time.
    """

    def test_method(self):
        workflow = load_workflow(workflow_path)
        calls = find_dispatch_calls_in_workflow(workflow)
        errors = []
        for call in calls:
            target_path = WORKFLOWS_DIR / call.target_workflow
            if not target_path.exists():
                continue

            target_workflow = load_workflow(target_path)
            for input_name, value in call.passed_inputs.items():
                if _is_expression(value):
                    continue
                options = get_choice_options(target_workflow, input_name)
                if options is None:
                    continue
                if value not in options:
                    errors.append(
                        f"step '{call.step_name}' passes literal "
                        f"'{value}' for '{input_name}' to "
                        f"'{call.target_workflow}', but allowed options are: "
                        f"{options}"
                    )

        if errors:
            self.fail("\n".join(errors))

    return test_method


def _workflow_name_to_test_suffix(workflow_path: Path) -> str:
    """Converts a workflow filename to a valid Python identifier suffix."""
    return workflow_path.stem.replace("-", "_").replace(".", "_")


# Dynamically generate test methods for workflow files that have dispatch calls.
for _workflow_path in sorted(WORKFLOWS_DIR.glob("*.yml")):
    _workflow = load_workflow(_workflow_path)
    if not find_dispatch_calls_in_workflow(_workflow):
        continue

    _suffix = _workflow_name_to_test_suffix(_workflow_path)

    _test = _make_unexpected_inputs_test(_workflow_path)
    _test.__doc__ = f"No unexpected dispatch inputs in {_workflow_path.name}"
    setattr(WorkflowDispatchInputsTest, f"test_no_unexpected_inputs__{_suffix}", _test)

    _test = _make_required_inputs_test(_workflow_path)
    _test.__doc__ = f"All required dispatch inputs passed in {_workflow_path.name}"
    setattr(
        WorkflowDispatchInputsTest, f"test_required_inputs_passed__{_suffix}", _test
    )

    _test = _make_literal_choice_values_test(_workflow_path)
    _test.__doc__ = (
        f"Literal dispatch values are valid for choice inputs in {_workflow_path.name}"
    )
    setattr(
        WorkflowDispatchInputsTest,
        f"test_literal_values_valid_for_choices__{_suffix}",
        _test,
    )


if __name__ == "__main__":
    unittest.main()
