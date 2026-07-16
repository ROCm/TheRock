# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared helpers for workflow YAML tests."""

import re
from pathlib import Path

import yaml

WORKFLOWS_DIR = Path(__file__).resolve().parents[3] / ".github" / "workflows"
_MATRIX_REFERENCE_RE = re.compile(r"\bmatrix\.([A-Za-z_][A-Za-z0-9_]*)")


def load_workflow(path: Path) -> dict:
    """Loads a YAML workflow file from the given Path as a JSON dictionary."""
    with open(path) as f:
        return yaml.safe_load(f)


def get_workflow_job(workflow: dict, job_name: str) -> dict:
    """Returns a workflow job definition.

    For a workflow with:
        jobs:
          build_wheels:
            uses: ./.github/workflows/build_wheels.yml
            with:
              python_version: ${{ matrix.python_version }}

    get_workflow_job(workflow, "build_wheels") returns the dictionary
    containing the uses/with blocks for that job.
    """
    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict):
        raise KeyError("workflow has no jobs block")

    job = jobs[job_name]
    if not isinstance(job, dict):
        raise KeyError(f"workflow job {job_name!r} is not a mapping")
    return job


def get_matrix_references(value: object) -> set[str]:
    """Extracts top-level matrix keys referenced by a workflow YAML value.

    For a workflow value with:
        with:
          python_version: ${{ matrix.python_version }}
          package_url: >-
            ${{
              format('{0}/{1}/index.html',
                  needs.build.outputs.package_find_links_url,
                  matrix.amdgpu_family)
            }}

    get_matrix_references(value) returns:
        {"python_version", "amdgpu_family"}

    Nested matrix objects like matrix.family_info.amdgpu_family return only the
    top-level matrix key, {"family_info"}.
    """
    if isinstance(value, str):
        return set(_MATRIX_REFERENCE_RE.findall(value))

    if isinstance(value, dict):
        references = set()
        for child_value in value.values():
            references.update(get_matrix_references(child_value))
        return references

    if isinstance(value, list):
        references = set()
        for child_value in value:
            references.update(get_matrix_references(child_value))
        return references

    return set()


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


def _get_dispatch_inputs(workflow: dict) -> dict:
    """Returns the workflow_dispatch inputs dict, or empty dict."""
    dispatch = _get_workflow_dispatch_block(workflow)
    if dispatch is None:
        return {}
    inputs = dispatch.get("inputs")
    if not isinstance(inputs, dict):
        return {}
    return inputs


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
    return set(_get_dispatch_inputs(workflow).keys())


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
    required = set()
    for name, props in _get_dispatch_inputs(workflow).items():
        if isinstance(props, dict):
            if props.get("required", False) and "default" not in props:
                required.add(name)
    return required


def get_transitive_workflow_uses(root_filenames: list[str]) -> set[str]:
    """Returns all workflow filenames transitively referenced via reusable workflow calls.

    Starting from the given root workflow filenames, follows all
    ``uses: ./.github/workflows/<name>.yml`` references in job definitions
    and returns the complete set of workflow filenames (including the roots).
    """
    visited: set[str] = set()
    queue = list(root_filenames)
    while queue:
        filename = queue.pop()
        if filename in visited:
            continue
        visited.add(filename)
        workflow_path = WORKFLOWS_DIR / filename
        if not workflow_path.exists():
            continue
        workflow = load_workflow(workflow_path)
        if not isinstance(workflow, dict):
            continue
        jobs = workflow.get("jobs")
        if not isinstance(jobs, dict):
            continue
        for job_def in jobs.values():
            if not isinstance(job_def, dict):
                continue
            uses = job_def.get("uses")
            if isinstance(uses, str) and uses.startswith("./.github/workflows/"):
                ref_filename = uses.removeprefix("./.github/workflows/")
                queue.append(ref_filename)
    return visited


def get_choice_options(workflow: dict, input_name: str) -> list | None:
    """Extracts the options list for a type: choice workflow_dispatch input.

    For a workflow with:
        on:
          workflow_dispatch:
            inputs:
              amdgpu_family:
                type: choice
                options:
                  - gfx94X-dcgpu
                  - gfx110X-all

    get_choice_options(workflow, "amdgpu_family") returns:
        ["gfx94X-dcgpu", "gfx110X-all"]

    Returns None if the input doesn't exist or isn't type: choice.
    """
    input_def = _get_dispatch_inputs(workflow).get(input_name)
    if not isinstance(input_def, dict):
        return None
    if input_def.get("type") != "choice":
        return None
    options = input_def.get("options")
    if not isinstance(options, list):
        return None
    return options
