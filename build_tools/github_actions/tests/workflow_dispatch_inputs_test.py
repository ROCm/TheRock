"""Validate that benc-uk/workflow-dispatch calls only pass inputs defined by target workflows.

The benc-uk/workflow-dispatch action triggers workflows via the GitHub REST API's
"Create a workflow dispatch event" endpoint. This endpoint rejects any inputs that
are not defined in the target workflow's `on: workflow_dispatch: inputs:` section.

This test catches mismatches at lint time rather than waiting for CI failures.
See: https://github.com/ROCm/TheRock/pull/2557 for an example of this class of bug.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import unittest

import yaml

WORKFLOWS_DIR = Path(__file__).resolve().parents[3] / ".github" / "workflows"

WORKFLOW_DISPATCH_ACTION_NAME = "benc-uk/workflow-dispatch"


def load_workflow(path: Path) -> dict:
    """Loads a workflow file from the given Path as a JSON dictionary."""
    with open(path) as f:
        return yaml.safe_load(f)


def get_workflow_dispatch_inputs(workflow: dict) -> set:
    """Extracts input names from a workflow's on.workflow_dispatch.inputs section."""
    # PyYAML parses the unquoted YAML key `on:` as boolean True.
    on_block = workflow.get("on") or workflow.get(True)
    if not isinstance(on_block, dict):
        return set()
    dispatch = on_block.get("workflow_dispatch")
    if not isinstance(dispatch, dict):
        return set()
    inputs = dispatch.get("inputs")
    if not isinstance(inputs, dict):
        return set()
    return set(inputs.keys())


def get_required_workflow_dispatch_inputs(workflow: dict) -> set:
    """Extracts required input names (no default) from workflow_dispatch."""
    # PyYAML parses the unquoted YAML key `on:` as boolean True.
    on_block = workflow.get("on") or workflow.get(True)
    if not isinstance(on_block, dict):
        return set()
    dispatch = on_block.get("workflow_dispatch")
    if not isinstance(dispatch, dict):
        return set()
    inputs_def = dispatch.get("inputs")
    if not isinstance(inputs_def, dict):
        return set()
    required = set()
    for name, props in inputs_def.items():
        if isinstance(props, dict):
            if props.get("required", False) and "default" not in props:
                required.add(name)
    return required


def parse_dispatch_inputs_json(inputs_raw: str) -> set:
    """Parses the JSON inputs string from a benc-uk/workflow-dispatch step."""
    if not inputs_raw:
        return set()

    parsed = json.loads(inputs_raw)
    if isinstance(parsed, dict):
        return set(parsed.keys())

    return set()


@dataclass
class DispatchCall:
    """A single benc-uk/workflow-dispatch action invocation."""

    step_name: str
    target_workflow: str
    passed_inputs: set


def find_dispatch_calls_in_workflow(workflow: dict) -> list[DispatchCall]:
    """Find benc-uk/workflow-dispatch steps in a single workflow."""
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
    """Verify benc-uk/workflow-dispatch calls only pass valid inputs.

    Test cases are generated dynamically, one per workflow file.
    """

    pass


def _make_unexpected_inputs_test(workflow_path: Path):
    """Create a test that checks for unexpected inputs in dispatch calls."""

    def test_method(self):
        workflow = load_workflow(workflow_path)
        calls = find_dispatch_calls_in_workflow(workflow)
        if not calls:
            return

        errors = []
        for call in calls:
            target_path = WORKFLOWS_DIR / call.target_workflow
            if not target_path.exists():
                errors.append(
                    f"step '{call.step_name}' dispatches "
                    f"'{call.target_workflow}' which does not exist"
                )
                continue

            target_workflow = load_workflow(target_path)
            accepted_inputs = get_workflow_dispatch_inputs(target_workflow)
            if not accepted_inputs:
                errors.append(
                    f"step '{call.step_name}' dispatches "
                    f"'{call.target_workflow}' which has no workflow_dispatch inputs"
                )
                continue

            unexpected = call.passed_inputs - accepted_inputs
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
    """Create a test that checks all required inputs are passed."""

    def test_method(self):
        workflow = load_workflow(workflow_path)
        calls = find_dispatch_calls_in_workflow(workflow)
        if not calls:
            return

        errors = []
        for call in calls:
            target_path = WORKFLOWS_DIR / call.target_workflow
            if not target_path.exists():
                continue

            target_workflow = load_workflow(target_path)
            required_inputs = get_required_workflow_dispatch_inputs(target_workflow)
            missing = required_inputs - call.passed_inputs
            if missing:
                errors.append(
                    f"step '{call.step_name}' does not pass required inputs to "
                    f"'{call.target_workflow}': {sorted(missing)}"
                )

        if errors:
            self.fail("\n".join(errors))

    return test_method


def _workflow_name_to_test_suffix(workflow_path: Path) -> str:
    """Convert a workflow filename to a valid Python identifier suffix."""
    return workflow_path.stem.replace("-", "_").replace(".", "_")


# Dynamically generate test methods, one per workflow file.
for _workflow_path in sorted(WORKFLOWS_DIR.glob("*.yml")):
    _suffix = _workflow_name_to_test_suffix(_workflow_path)

    _test = _make_unexpected_inputs_test(_workflow_path)
    _test.__doc__ = f"No unexpected dispatch inputs in {_workflow_path.name}"
    setattr(WorkflowDispatchInputsTest, f"test_no_unexpected_inputs__{_suffix}", _test)

    _test = _make_required_inputs_test(_workflow_path)
    _test.__doc__ = f"All required dispatch inputs passed in {_workflow_path.name}"
    setattr(
        WorkflowDispatchInputsTest, f"test_required_inputs_passed__{_suffix}", _test
    )


if __name__ == "__main__":
    unittest.main()
