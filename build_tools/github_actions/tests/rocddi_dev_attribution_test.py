# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Render the production workflow's source, suite and matrix attribution."""

import ast
import json
import re
import subprocess
from pathlib import Path

import pytest

from rocddi_dev_authorization_test import _evaluate
from workflow_utils import WORKFLOWS_DIR, load_workflow


def render(template, context):
    def replacement(match):
        expression = match[1].replace("&&", " and ").replace("||", " or ")
        value = _evaluate(ast.parse(expression.strip(), mode="eval").body, context)
        return "" if value is None else str(value)

    return re.sub(r"\$\{\{(.*?)\}\}", replacement, template, flags=re.DOTALL)


@pytest.fixture(scope="module")
def workflow():
    return load_workflow(WORKFLOWS_DIR / "rocddi_cts_dev.yml")


def dispatch_context(workflow, **inputs):
    defaults = {
        f"inputs.{name}": value.get("default", "")
        for name, value in workflow[True]["workflow_dispatch"]["inputs"].items()
    }
    return (
        defaults
        | {
            "github.event_name": "workflow_dispatch",
            "github.sha": "a" * 40,
        }
        | {f"inputs.{name}": value for name, value in inputs.items()}
    )


@pytest.mark.parametrize(
    "entries, expected",
    [
        ([], "host"),
        (["gfx90a"], "gfx90a"),
        (["gfx90a", "gfx950-dcgpu"], "gfx90a + more"),
        (
            [
                {
                    "amdgpu_family": "gfx110X-all",
                    "amdgpu_targets": ["gfx1100"],
                    "test_runs_on": ["self-hosted", "my-gfx1100"],
                }
            ],
            "gfx110X-all",
        ),
        (
            [{"amdgpu_family": "gfx110X-all"}, {"amdgpu_family": "gfx90a"}],
            "gfx110X-all + more",
        ),
        (["gfx90a", {"amdgpu_family": "gfx110X-all"}], "gfx90a + more"),
        ([{"amdgpu_family": "gfx110X-all"}, "gfx90a"], "gfx110X-all + more"),
    ],
)
def test_manual_titles_summarize_string_object_and_mixed_matrices(
    workflow, entries, expected
):
    context = dispatch_context(workflow, arch_matrix=json.dumps(entries))
    assert render(workflow["run-name"], context) == (
        f"Rocddi shared/rocddi-cts-dev | both | {expected}"
    )


@pytest.mark.parametrize("suites", ["cts", "hip", "both"])
def test_runtime_refs_and_suites_distinguish_runs_at_the_same_therock_head(
    workflow, suites
):
    before = dispatch_context(
        workflow, rocm_systems_ref="b" * 40, suites=suites, arch_matrix='["gfx90a"]'
    )
    after = before | {"inputs.rocm_systems_ref": "c" * 40}
    assert before["github.sha"] == after["github.sha"]
    assert (
        render(workflow["run-name"], before) == f"Rocddi {'b' * 40} | {suites} | gfx90a"
    )
    assert (
        render(workflow["run-name"], after) == f"Rocddi {'c' * 40} | {suites} | gfx90a"
    )


def test_push_title_matches_production_host_cts_inputs(workflow):
    context = {
        "github.event_name": "push",
        "inputs.rocm_systems_ref": "",
        "inputs.arch_matrix": "",
        "inputs.suites": "",
    }
    assert (
        render(workflow["run-name"], context)
        == "Rocddi shared/rocddi-cts-dev | cts | host"
    )
    step = next(
        step
        for step in workflow["jobs"]["configure"]["steps"]
        if step.get("id") == "configure"
    )
    assert {
        name: render(step["env"][name], context)
        for name in ("ROCM_SYSTEMS_REF", "ARCH_MATRIX", "SUITES")
    } == {
        "ROCM_SYSTEMS_REF": "shared/rocddi-cts-dev",
        "ARCH_MATRIX": "[]",
        "SUITES": "cts",
    }


def test_manual_configure_receives_the_complete_requested_matrix_and_ref(workflow):
    entries = [
        "gfx90a",
        {
            "amdgpu_family": "gfx110X-all",
            "amdgpu_targets": ["gfx1100"],
            "test_runs_on": ["self-hosted", "my-gfx1100"],
        },
    ]
    matrix = json.dumps(entries)
    context = dispatch_context(
        workflow,
        rocm_systems_ref="users/example/fix-loader",
        arch_matrix=matrix,
        suites="hip",
    )
    step = next(
        step
        for step in workflow["jobs"]["configure"]["steps"]
        if step.get("id") == "configure"
    )
    assert {
        name: render(step["env"][name], context)
        for name in ("ROCM_SYSTEMS_REF", "ARCH_MATRIX", "SUITES")
    } == {
        "ROCM_SYSTEMS_REF": "users/example/fix-loader",
        "ARCH_MATRIX": matrix,
        "SUITES": "hip",
    }


@pytest.mark.parametrize("repeat", ["1", "10000", "0", "1.5", ""])
def test_manual_repeat_is_forwarded_without_replacing_invalid_values(workflow, repeat):
    context = dispatch_context(workflow, hip_test_repeat=repeat)
    assert (
        render(workflow["jobs"]["test"]["with"]["hip_test_repeat"], context) == repeat
    )


def test_push_repeat_uses_the_single_execution_default(workflow):
    context = {"github.event_name": "push", "inputs.hip_test_repeat": ""}
    assert render(workflow["jobs"]["test"]["with"]["hip_test_repeat"], context) == "1"


@pytest.mark.parametrize(
    "repeat, expected", [(None, "1"), ("37", "37"), ("", ""), ("0", "0")]
)
def test_candidate_source_records_default_or_requested_repeat_count(
    tmp_path, monkeypatch, repeat, expected
):
    build_script = (
        WORKFLOWS_DIR.parents[1] / "build_tools/github_actions/rocddi_dev_build.sh"
    )
    source = (
        build_script.read_text().split("python - <<'PY'\n", 1)[1].split("\nPY", 1)[0]
    )
    commit = "a" * 40
    values = {
        "EXTERNAL_REPO_CONFIG": json.dumps({"ref": commit}),
        "ROCM_SYSTEMS_REF": "shared/rocddi-cts-dev",
        "BASELINE_RUN_ID": "123",
        "AMDGPU_FAMILY": "",
        "AMDGPU_TARGETS": "",
        "TEST_RUNS_ON": '""',
        "HIP_TEST_LABEL": "all",
        "HIP_TEST_FILTER": "",
        "HIP_TEST_LEVEL": "level_0",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    if repeat is None:
        monkeypatch.delenv("HIP_TEST_REPEAT", raising=False)
    else:
        monkeypatch.setenv("HIP_TEST_REPEAT", repeat)
    monkeypatch.chdir(tmp_path)
    outputs = iter([commit, commit, "inert compiler version"])
    monkeypatch.setattr(
        subprocess, "check_output", lambda *args, **kwargs: next(outputs)
    )
    exec(source, {})
    assert json.loads(Path("SOURCE.json").read_text())["hip_test_repeat"] == expected
