# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate runtime development matrices and the workflow HIP selection command."""

import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import configure_rocddi_dev as configure
from hiptests_runner_test import installed_ctest_prefix
from workflow_utils import WORKFLOWS_DIR, load_workflow

RUNNERS = {
    "cdna2": {"linux": {"family": "gfx90a", "test-runs-on": "cdna2-runner"}},
    "cdna4": {
        "linux": {
            "family": "gfx950-dcgpu",
            "test-runs-on-labels": [{"label": "cdna4-runner", "count": 1}],
        }
    },
}


def matrix(entries, *, run_hip=True):
    return configure.configure_matrix(
        json.dumps(entries), run_hip=run_hip, runner_families=RUNNERS
    )["include"]


def test_multiple_architectures_use_real_cmake_targets_and_configured_runners():
    rows = matrix(["gfx90a", "gfx950-dcgpu"])
    assert [row["amdgpu_targets"] for row in rows] == ["gfx90a", "gfx950"]
    assert [json.loads(row["test_runs_on"]) for row in rows] == [
        "cdna2-runner",
        "cdna4-runner",
    ]
    assert len({row["profile"] for row in rows}) == 2


def test_explicit_target_subset_and_runner_labels_are_preserved():
    row = matrix(
        [
            {
                "amdgpu_family": "gfx110X-all",
                "amdgpu_targets": ["gfx1100", "gfx1100"],
                "test_runs_on": ["self-hosted", "my-gfx1100"],
            }
        ]
    )[0]
    assert row["amdgpu_targets"] == "gfx1100"
    assert json.loads(row["test_runs_on"]) == ["self-hosted", "my-gfx1100"]


def test_explicit_runner_supports_target_without_default_hardware():
    row = matrix([{"amdgpu_family": "gfx908", "test_runs_on": "reserved-mi100"}])[0]
    assert row["amdgpu_targets"] == "gfx908"


def test_target_features_are_preserved():
    row = matrix([{"amdgpu_family": "gfx90a", "amdgpu_targets": "gfx90a:xnack+"}])[0]
    assert row["amdgpu_targets"] == "gfx90a:xnack+"


def test_native_matrix_uses_existing_device_and_isolation_options():
    options = matrix(["gfx90a"])[0]["container_options"]
    assert "--user 0:0" in options
    assert "--device /dev/kfd" in options
    assert "--ulimit memlock=-1:-1" in options
    assert "--group-add" in options
    assert "--env-file /etc/podinfo/gha-gpu-isolation-settings" in options


def test_apptainer_matrix_uses_existing_backend_user_policy():
    options = matrix(
        [{"amdgpu_family": "gfx125X-dcgpu", "test_runs_on": "apptainer-gpu"}]
    )[0]["container_options"]
    assert "--user 0:0" not in options
    assert "--device /dev/kfd" in options


@pytest.mark.parametrize(
    "entries, message",
    [
        ({"include": []}, "JSON array"),
        ([123], "Invalid arch_matrix"),
        ([{"family": "gfx90a"}], "Invalid arch_matrix"),
        (["gfx-typo"], "Unknown AMDGPU"),
        (["gfx908"], "supply test_runs_on"),
        ([{"amdgpu_family": "gfx90a", "test_runs_on": []}], "No Linux test runner"),
        ([{"amdgpu_family": "gfx90a", "test_runs_on": "\n"}], "No Linux test runner"),
        ([{"amdgpu_family": "gfx90a", "amdgpu_targets": []}], "nonempty"),
        ([{"amdgpu_family": "gfx90a", "amdgpu_targets": [42]}], "does not belong"),
        (
            [{"amdgpu_family": "gfx90a", "amdgpu_targets": ["gfx950"]}],
            "does not belong",
        ),
        (
            [{"amdgpu_family": "gfx90a", "amdgpu_targets": "gfx90a;bad"}],
            "does not belong",
        ),
        (["gfx90a"] * 257, "256-job limit"),
    ],
)
def test_invalid_or_unroutable_entries_fail_without_dropping_rows(entries, message):
    with pytest.raises(ValueError, match=message):
        matrix(entries)


def test_empty_matrix_is_explicit_host_cts_only():
    assert matrix([], run_hip=False)[0]["profile"] == "host"
    with pytest.raises(ValueError, match="nonempty arch_matrix"):
        matrix([], run_hip=True)


def test_branch_is_resolved_once_for_the_complete_matrix(tmp_path, monkeypatch):
    output = tmp_path / "outputs"
    summary = tmp_path / "summary"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    commit = "a" * 40
    with (
        mock.patch.object(
            configure, "get_all_families_for_trigger_types", return_value=RUNNERS
        ),
        mock.patch.object(
            configure, "gha_resolve_git_ref", return_value=commit
        ) as resolve,
    ):
        configure.main(
            [
                "--rocm-systems-ref",
                "users/example/fix-loader",
                "--arch-matrix",
                '["gfx90a", "gfx950-dcgpu"]',
            ]
        )
    resolve.assert_called_once_with("ROCm/rocm-systems", "users/example/fix-loader")
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    source = json.loads(values["external_repo_config"])
    assert source["repository"] == "ROCm/rocm-systems"
    assert source["ref"] == commit
    assert source["submodule_path"] == "rocm-systems"
    assert len(json.loads(values["matrix"])["include"]) == 2
    assert values["run_cts"] == values["run_hip"] == "true"
    attribution = json.loads(summary.read_text().split("```json\n")[1].split("```")[0])
    assert attribution == {
        "rocm_systems_ref": "users/example/fix-loader",
        "rocm_systems_commit": commit,
        "suites": "both",
        "matrix": json.loads(values["matrix"]),
    }


def test_default_revision_is_shared_runtime_branch(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "outputs"))
    with (
        mock.patch.object(
            configure, "get_all_families_for_trigger_types", return_value=RUNNERS
        ),
        mock.patch.object(
            configure, "gha_resolve_git_ref", return_value="b" * 40
        ) as resolve,
    ):
        configure.main(["--suites", "cts"])
    resolve.assert_called_once_with("ROCm/rocm-systems", "shared/rocddi-cts-dev")


def test_repeat_input_defaults_and_forwarding():
    caller = load_workflow(WORKFLOWS_DIR / "rocddi_cts_dev.yml")
    reusable = load_workflow(WORKFLOWS_DIR / "rocddi_dev_test.yml")
    for inputs in (
        (caller.get("on") or caller[True])["workflow_dispatch"]["inputs"],
        (reusable.get("on") or reusable[True])["workflow_call"]["inputs"],
    ):
        assert inputs["hip_test_repeat"]["type"] == "string"
        assert inputs["hip_test_repeat"]["default"] == "1"
        assert not inputs["hip_test_repeat"].get("required", False)
    assert caller["jobs"]["test"]["with"]["hip_test_repeat"] == (
        "${{ github.event_name == 'push' && '1' || inputs.hip_test_repeat }}"
    )
    step = next(
        step
        for step in reusable["jobs"]["hip"]["steps"]
        if step.get("name") == "Run selected upstream HIP tests"
    )
    assert step["env"]["HIP_TEST_REPEAT"] == "${{ inputs.hip_test_repeat }}"
    assert reusable["jobs"]["build"]["env"]["HIP_TEST_REPEAT"] == (
        "${{ inputs.hip_test_repeat }}"
    )


def test_debugger_is_optional_and_installed_only_in_the_hip_job():
    caller = load_workflow(WORKFLOWS_DIR / "rocddi_cts_dev.yml")
    reusable = load_workflow(WORKFLOWS_DIR / "rocddi_dev_test.yml")
    for inputs in (
        (caller.get("on") or caller[True])["workflow_dispatch"]["inputs"],
        (reusable.get("on") or reusable[True])["workflow_call"]["inputs"],
    ):
        assert inputs["hip_gdb_on_failure"]["type"] == "boolean"
        assert inputs["hip_gdb_on_failure"]["default"] is False
        assert not inputs["hip_gdb_on_failure"].get("required", False)
    assert caller["jobs"]["test"]["with"]["hip_gdb_on_failure"] == (
        "${{ inputs.hip_gdb_on_failure || false }}"
    )
    installs = [
        (job, step)
        for job, config in reusable["jobs"].items()
        for step in config["steps"]
        if "apt-get install" in step.get("run", "")
    ]
    assert len(installs) == 1 and installs[0][0] == "hip"
    assert installs[0][1]["if"] == "${{ inputs.hip_gdb_on_failure }}"
    run = next(
        step
        for step in reusable["jobs"]["hip"]["steps"]
        if step.get("name") == "Run selected upstream HIP tests"
    )
    assert not run.get("continue-on-error", False)
    assert run["env"]["HIP_GDB_ON_FAILURE"] == "${{ inputs.hip_gdb_on_failure }}"


@pytest.mark.parametrize(
    "test_filter, repeat, returncode",
    [
        ("Unit_Unlabelled", "1", 0),
        ("Unit_Unlabelled", "3", 0),
        ("Unit_Fail", "3", 1),
        ("Unit_Unlabelled", "0", 2),
        ("Unit_Unlabelled", "$(touch repeat-injected)", 2),
    ],
)
def test_workflow_preserves_selection_and_validates_quoted_repeat(
    installed_ctest_prefix, tmp_path, test_filter, repeat, returncode
):
    root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load(
        (root / ".github/workflows/rocddi_dev_test.yml").read_text()
    )
    command = next(
        step["run"]
        for step in workflow["jobs"]["hip"]["steps"]
        if step.get("name") == "Run selected upstream HIP tests"
    )
    tests = installed_ctest_prefix / "share/hip/catch_tests/CTestTestfile.cmake"
    with tests.open("a") as output:
        output.write(f'add_test(Unit_Unlabelled "{shutil.which("cmake")}" -E true)\n')
    (tmp_path / "candidate").symlink_to(
        installed_ctest_prefix, target_is_directory=True
    )
    (tmp_path / "build_tools").symlink_to(
        root / "build_tools", target_is_directory=True
    )
    (tmp_path / "diagnostics").mkdir()
    run = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", command],
        cwd=tmp_path,
        env=os.environ
        | {
            "THEROCK_BIN_DIR": str(installed_ctest_prefix / "bin"),
            "AMDGPU_FAMILIES": "gfx90a",
            "HIP_TEST_LABEL": "all",
            "HIP_TEST_FILTER": test_filter,
            "HIP_TEST_REPEAT": repeat,
        },
        text=True,
        capture_output=True,
    )
    assert run.returncode == returncode, run.stdout + run.stderr
    report = tmp_path / "diagnostics/hip-tests.xml"
    assert not (tmp_path / "repeat-injected").exists()
    if returncode == 2:
        assert "--repeat: must be an integer from 1 to 10000" in run.stdout
        assert not report.exists()
        return
    assert [case.get("name") for case in ET.parse(report).iter("testcase")] == [
        test_filter
    ]
