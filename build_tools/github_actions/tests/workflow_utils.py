"""Shared helpers for workflow YAML tests."""

from pathlib import Path

import yaml

WORKFLOWS_DIR = Path(__file__).resolve().parents[3] / ".github" / "workflows"


def load_workflow(path: Path) -> dict:
    """Loads a YAML workflow file from the given Path as a JSON dictionary."""
    with open(path) as f:
        return yaml.safe_load(f)


def _get_workflow_dispatch_block(workflow: dict) -> dict | None:
    """Returns the workflow_dispatch block, or None."""
    # PyYAML parses the unquoted YAML key `on:` as boolean True.
    on_block = workflow.get("on") or workflow.get(True)
    if not isinstance(on_block, dict):
        return None
    dispatch = on_block.get("workflow_dispatch")
    if not isinstance(dispatch, dict):
        return None
    return dispatch


def get_workflow_dispatch_inputs(workflow: dict) -> set:
    """Extracts input names from a workflow's on.workflow_dispatch.inputs section.

    For a workflow with:
        on:
          workflow_dispatch:
            inputs:
              amdgpu_family: ...
              release_type: ...

    Returns: {"amdgpu_family", "release_type"}
    """
    dispatch = _get_workflow_dispatch_block(workflow)
    if dispatch is None:
        return set()
    inputs = dispatch.get("inputs")
    if not isinstance(inputs, dict):
        return set()
    return set(inputs.keys())


def get_required_workflow_dispatch_inputs(workflow: dict) -> set:
    """Extracts required input names (no default) from workflow_dispatch.

    For a workflow with:
        on:
          workflow_dispatch:
            inputs:
              amdgpu_family:
                required: true
              release_type:
                required: true
                default: dev

    Returns: {"amdgpu_family"}  (release_type has a default)
    """
    dispatch = _get_workflow_dispatch_block(workflow)
    if dispatch is None:
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


def get_choice_options(workflow: dict, input_name: str) -> list | None:
    """Extracts the options list for a type: choice workflow_dispatch input.

    Returns None if the input doesn't exist or isn't type: choice.
    Returns the list of option strings if it is.
    """
    dispatch = _get_workflow_dispatch_block(workflow)
    if dispatch is None:
        return None
    inputs = dispatch.get("inputs")
    if not isinstance(inputs, dict):
        return None
    input_def = inputs.get(input_name)
    if not isinstance(input_def, dict):
        return None
    if input_def.get("type") != "choice":
        return None
    options = input_def.get("options")
    if not isinstance(options, list):
        return None
    return options
