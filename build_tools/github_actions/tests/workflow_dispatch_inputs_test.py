"""Validate that benc-uk/workflow-dispatch calls only pass inputs defined by target workflows.

The benc-uk/workflow-dispatch action triggers workflows via the GitHub REST API's
"Create a workflow dispatch event" endpoint. This endpoint rejects any inputs that
are not defined in the target workflow's `on: workflow_dispatch: inputs:` section.

This test catches mismatches at lint time rather than waiting for CI failures.
See: https://github.com/ROCm/TheRock/pull/2557 for an example of this class of bug.
"""

import json
from pathlib import Path
import re
import unittest

import yaml

WORKFLOWS_DIR = Path(__file__).resolve().parents[3] / ".github" / "workflows"

WORKFLOW_DISPATCH_ACTION = "benc-uk/workflow-dispatch@"


def load_workflow(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_workflow_dispatch_inputs(workflow: dict) -> set:
    """Extract input names from a workflow's on.workflow_dispatch.inputs section."""
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


def find_workflow_dispatch_calls(workflows_dir: Path) -> list:
    """Find all steps using benc-uk/workflow-dispatch and extract their inputs.

    Returns a list of dicts with keys:
        - caller: filename of the calling workflow
        - step_name: name of the step
        - target_workflow: filename of the target workflow
        - passed_inputs: set of input names passed in the 'inputs' field
    """
    calls = []
    for workflow_path in sorted(workflows_dir.glob("*.yml")):
        workflow = load_workflow(workflow_path)
        if not workflow or "jobs" not in workflow:
            continue
        for job_name, job in workflow["jobs"].items():
            steps = job.get("steps", [])
            for step in steps:
                uses = step.get("uses", "")
                if WORKFLOW_DISPATCH_ACTION not in uses:
                    continue
                with_block = step.get("with", {})
                target = with_block.get("workflow", "")
                inputs_raw = with_block.get("inputs", "")

                # Parse the JSON inputs string.
                # The inputs field is a JSON object as a string. GitHub
                # expressions (${{ ... }}) are inside JSON string values,
                # so they don't affect JSON structure parsing.
                passed_inputs = set()
                if inputs_raw:
                    try:
                        parsed = json.loads(inputs_raw)
                        if isinstance(parsed, dict):
                            passed_inputs = set(parsed.keys())
                    except json.JSONDecodeError:
                        # Expressions containing quotes can break JSON parsing.
                        # Replace ${{ ... }} with placeholders to extract keys.
                        sanitized = re.sub(
                            r"\$\{\{.*?\}\}", "__placeholder__", inputs_raw
                        )
                        try:
                            parsed = json.loads(sanitized)
                            if isinstance(parsed, dict):
                                passed_inputs = set(parsed.keys())
                        except json.JSONDecodeError as e:
                            # Still can't parse - treat as error
                            print(
                                f"  WARNING: Cannot parse inputs JSON in "
                                f"'{step.get('name', '(unnamed)')}': {e}"
                            )

                calls.append(
                    {
                        "caller": workflow_path.name,
                        "step_name": step.get("name", "(unnamed)"),
                        "target_workflow": target,
                        "passed_inputs": passed_inputs,
                    }
                )
    return calls


class WorkflowDispatchInputsTest(unittest.TestCase):
    """Verify benc-uk/workflow-dispatch calls only pass valid inputs."""

    def test_all_dispatched_inputs_exist_in_target_workflow(self):
        """Each input passed via benc-uk/workflow-dispatch must be defined in
        the target workflow's on.workflow_dispatch.inputs section."""
        calls = find_workflow_dispatch_calls(WORKFLOWS_DIR)
        self.assertTrue(len(calls) > 0, "Expected to find workflow-dispatch calls")

        print(f"\nValidating {len(calls)} benc-uk/workflow-dispatch call(s):")

        errors = []
        for call in calls:
            target_path = WORKFLOWS_DIR / call["target_workflow"]
            if not target_path.exists():
                errors.append(
                    f"{call['caller']}: step '{call['step_name']}' dispatches "
                    f"'{call['target_workflow']}' which does not exist"
                )
                continue

            target_workflow = load_workflow(target_path)
            accepted_inputs = get_workflow_dispatch_inputs(target_workflow)

            if not accepted_inputs:
                errors.append(
                    f"{call['caller']}: step '{call['step_name']}' dispatches "
                    f"'{call['target_workflow']}' which has no workflow_dispatch inputs"
                )
                continue

            unexpected = call["passed_inputs"] - accepted_inputs
            if unexpected:
                errors.append(
                    f"{call['caller']}: step '{call['step_name']}' passes unexpected "
                    f"inputs to '{call['target_workflow']}': {sorted(unexpected)}. "
                    f"Accepted inputs: {sorted(accepted_inputs)}"
                )
                print(f"  FAIL {call['caller']} -> {call['target_workflow']}")
                print(f"       step: '{call['step_name']}'")
                print(f"       unexpected inputs: {sorted(unexpected)}")
                print(f"       accepted inputs:   {sorted(accepted_inputs)}")
            else:
                print(f"  OK   {call['caller']} -> {call['target_workflow']}")
                print(f"       step: '{call['step_name']}'")
                print(f"       inputs: {sorted(call['passed_inputs'])}")

        if errors:
            self.fail(
                "workflow-dispatch input validation failures:\n  " + "\n  ".join(errors)
            )

    def test_all_required_target_inputs_are_passed(self):
        """Each required input (no default) in the target workflow's
        workflow_dispatch section should be passed by the caller."""
        calls = find_workflow_dispatch_calls(WORKFLOWS_DIR)

        print(f"\nChecking required inputs for {len(calls)} dispatch call(s):")

        errors = []
        for call in calls:
            target_path = WORKFLOWS_DIR / call["target_workflow"]
            if not target_path.exists():
                continue

            target_workflow = load_workflow(target_path)
            on_block = target_workflow.get("on") or target_workflow.get(True)
            if not isinstance(on_block, dict):
                continue
            dispatch = on_block.get("workflow_dispatch")
            if not isinstance(dispatch, dict):
                continue
            inputs_def = dispatch.get("inputs")
            if not isinstance(inputs_def, dict):
                continue

            required_inputs = set()
            for name, props in inputs_def.items():
                if isinstance(props, dict):
                    if props.get("required", False) and "default" not in props:
                        required_inputs.add(name)

            missing = required_inputs - call["passed_inputs"]
            if missing:
                errors.append(
                    f"{call['caller']}: step '{call['step_name']}' does not pass "
                    f"required inputs to '{call['target_workflow']}': {sorted(missing)}"
                )
                print(f"  FAIL {call['caller']} -> {call['target_workflow']}")
                print(f"       step: '{call['step_name']}'")
                print(f"       missing required: {sorted(missing)}")
            elif required_inputs:
                print(f"  OK   {call['caller']} -> {call['target_workflow']}")
                print(f"       all required inputs passed: {sorted(required_inputs)}")

        if errors:
            self.fail("Missing required inputs:\n  " + "\n  ".join(errors))


if __name__ == "__main__":
    unittest.main()
