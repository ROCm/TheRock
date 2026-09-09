# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared helpers for workflow YAML tests."""

from dataclasses import dataclass
import json
import re
from pathlib import Path

import yaml

WORKFLOWS_DIR = Path(__file__).resolve().parents[3] / ".github" / "workflows"
_MATRIX_REFERENCE_RE = re.compile(r"\bmatrix\.([A-Za-z_][A-Za-z0-9_]*)")
_LOCAL_WORKFLOW_PREFIX = "./.github/workflows/"
_WORKFLOW_DISPATCH_ACTION_NAME = "benc-uk/workflow-dispatch"


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


def _get_workflow_call_block(workflow: dict) -> dict | None:
    """Returns the workflow_call block, or None."""
    # PyYAML parses the unquoted YAML key `on:` as boolean True.
    on_block = workflow.get("on") or workflow.get(True)
    if not isinstance(on_block, dict):
        return None
    call = on_block.get("workflow_call")
    if not isinstance(call, dict):
        return None
    return call


def get_workflow_call_inputs(workflow: dict) -> set:
    """Extracts input names from a workflow's on.workflow_call.inputs section.

    For a workflow with:
        on:
          workflow_call:
            inputs:
              build_config: ...
              quartz_tracking_id: ...

    Returns: {"build_config", "quartz_tracking_id"}
    """
    call = _get_workflow_call_block(workflow)
    if call is None:
        return set()
    inputs = call.get("inputs")
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
    required = set()
    for name, props in _get_dispatch_inputs(workflow).items():
        if isinstance(props, dict):
            if props.get("required", False) and "default" not in props:
                required.add(name)
    return required


def get_transitive_workflow_uses(root_filenames: list[str]) -> set[str]:
    """Returns all workflow filenames transitively referenced via reusable workflow calls.

    Starting from the given root workflow filenames, follows all
    `uses: ./.github/workflows/<name>.yml` references in job definitions
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
        for edge in get_workflow_edges(load_workflow(workflow_path), filename):
            if edge.kind == "reusable":
                queue.append(edge.target)
    return visited


@dataclass
class WorkflowEdge:
    """A single propagation edge from one workflow file to another.

    `kind` is "reusable" for a `uses: ./.github/workflows/<name>.yml` job or
    "dispatch" for a `benc-uk/workflow-dispatch` step. `passed_inputs` is the
    set of input names forwarded across the edge (the `with:` keys for reusable
    jobs, the parsed JSON keys for dispatch steps).
    """

    source: str
    kind: str
    label: str
    target: str
    passed_inputs: set


def _parse_dispatch_inputs_keys(inputs_raw: str) -> set:
    """Returns the input names in a benc-uk/workflow-dispatch inputs JSON string."""
    if not inputs_raw:
        return set()
    parsed = json.loads(inputs_raw)
    if isinstance(parsed, dict):
        return set(parsed.keys())
    return set()


def get_workflow_edges(workflow: dict, source: str) -> list[WorkflowEdge]:
    """Returns all reusable and dispatch edges out of a single workflow."""
    jobs = workflow.get("jobs") if isinstance(workflow, dict) else None
    if not isinstance(jobs, dict):
        return []

    edges: list[WorkflowEdge] = []
    for job_name, job in jobs.items():
        if not isinstance(job, dict):
            continue

        uses = job.get("uses")
        if isinstance(uses, str) and uses.startswith(_LOCAL_WORKFLOW_PREFIX):
            with_block = job.get("with")
            passed = set(with_block.keys()) if isinstance(with_block, dict) else set()
            edges.append(
                WorkflowEdge(
                    source=source,
                    kind="reusable",
                    label=job_name,
                    target=uses.removeprefix(_LOCAL_WORKFLOW_PREFIX),
                    passed_inputs=passed,
                )
            )

        for step in job.get("steps", []):
            if not isinstance(step, dict):
                continue
            if _WORKFLOW_DISPATCH_ACTION_NAME not in step.get("uses", ""):
                continue
            with_block = step.get("with", {})
            edges.append(
                WorkflowEdge(
                    source=source,
                    kind="dispatch",
                    label=step.get("name", "(unnamed)"),
                    target=with_block.get("workflow", ""),
                    passed_inputs=_parse_dispatch_inputs_keys(
                        with_block.get("inputs", "")
                    ),
                )
            )
    return edges


def get_transitive_workflow_graph(
    root_filenames: list[str],
) -> tuple[dict[str, set], list[WorkflowEdge]]:
    """Walks the workflow call graph from the given roots, following both edges.

    Like `get_transitive_workflow_uses` but also follows
    `benc-uk/workflow-dispatch` steps, so it captures workflows triggered
    asynchronously as well as reusable-workflow calls.

    Returns a `(nodes, edges)` tuple where `nodes` maps each reachable
    workflow filename to the set of edge kinds used to reach it ("reusable"
    and/or "dispatch"; the roots map to an empty set), and `edges` is the flat
    list of every edge discovered during the walk.
    """
    nodes: dict[str, set] = {filename: set() for filename in root_filenames}
    edges: list[WorkflowEdge] = []
    queue = list(root_filenames)
    while queue:
        source = queue.pop()
        workflow_path = WORKFLOWS_DIR / source
        if not workflow_path.exists():
            continue
        for edge in get_workflow_edges(load_workflow(workflow_path), source):
            edges.append(edge)
            if not edge.target:
                continue
            if edge.target not in nodes:
                nodes[edge.target] = set()
                queue.append(edge.target)
            nodes[edge.target].add(edge.kind)
    return nodes, edges


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
