# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from workflow_utils import WORKFLOWS_DIR, load_workflow

SOURCE_ROOT = WORKFLOWS_DIR.parents[1]
GENERATOR_PATH = Path("build_tools/github_actions/fetch_test_configurations.py")
sys.path.insert(0, str(SOURCE_ROOT / "build_tools"))

from fetch_artifacts import S3Backend, WorkflowOutputRoot
import install_rocm_from_artifacts as installer


def _run_workflow_configure_step(
    source: Path, output: Path, scope: str, **env: str
) -> subprocess.CompletedProcess[str]:
    configure = load_workflow(WORKFLOWS_DIR / "test_artifacts.yml")["jobs"][
        "configure_test_matrix"
    ]
    step = next(step for step in configure["steps"] if step.get("id") == "configure")
    assert step["shell"] == "python"
    assert step["env"]["TEST_SCOPE"] == "${{ inputs.test_scope }}"
    assert step["env"]["SANITY_CHECK_ONLY_FOR_FAMILY"] == (
        "${{ inputs.sanity_check_only_for_family }}"
    )
    assert step["env"]["GENERIC_TEST_LABEL_OVERRIDES"] == (
        "${{ inputs.generic_test_label_overrides }}"
    )
    return subprocess.run(
        [sys.executable, "-c", step["run"]],
        cwd=source,
        env={
            **os.environ,
            "AMDGPU_FAMILIES": "",
            "AMDGPU_TARGETS": "",
            "BUILD_VARIANT": "release",
            "CI_CONFIG_PATH": "",
            "GENERIC_TEST_LABEL_OVERRIDES": "",
            "GITHUB_OUTPUT": str(output),
            "PROJECTS_TO_TEST": "rocgdb-cpu",
            "SANITY_CHECK_ONLY_FOR_FAMILY": "false",
            "SKIP_STAGES": "",
            "TEST_LABELS": "[]",
            "TEST_PLATFORM": "linux",
            "TEST_RUNS_ON": "",
            "TEST_RUNS_ON_CPU": "linux-cpu",
            "TEST_SCOPE": scope,
            "TEST_TYPE": "standard",
            **env,
        },
        text=True,
        capture_output=True,
        timeout=30,
    )


@pytest.mark.parametrize("scope", ["all", "family", "generic"])
def test_workflow_checks_explicit_scope_against_checked_out_generator(
    tmp_path: Path, scope: str
) -> None:
    generator = tmp_path / GENERATOR_PATH
    generator.parent.mkdir(parents=True)
    # Model a checkout with the legacy CLI, which ignores unknown arguments.
    generator.write_text(
        "import argparse, os\n"
        "from pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--platform')\n"
        "parser.parse_known_args()\n"
        "Path(os.environ['GITHUB_OUTPUT']).write_text('platform=linux\\n')\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    result = _run_workflow_configure_step(tmp_path, output, scope)
    if scope == "all":
        assert result.returncode == 0, result.stderr
        assert output.read_text(encoding="utf-8") == "platform=linux\n"
    else:
        assert result.returncode != 0
        assert f"does not support test_scope={scope!r}" in result.stderr
        assert not output.exists()


@pytest.mark.parametrize(
    "family, error",
    [
        (" ", "requires a non-empty AMDGPU_FAMILIES"),
        ("generic", "cannot use AMDGPU_FAMILIES=generic"),
    ],
    ids=["blank", "generic"],
)
def test_workflow_rejects_invalid_family_before_emitting_matrix(
    tmp_path: Path, family: str, error: str
) -> None:
    output = tmp_path / "output"
    result = _run_workflow_configure_step(
        SOURCE_ROOT,
        output,
        "family",
        AMDGPU_FAMILIES=family,
        PROJECTS_TO_TEST="*",
        TEST_LABELS='["test:miopen-dbsync"]',
    )
    assert result.returncode != 0
    assert error in result.stderr
    assert not output.exists()


def test_workflow_rejects_generic_sanity_only_before_emitting_matrix(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    result = _run_workflow_configure_step(
        SOURCE_ROOT, output, "generic", SANITY_CHECK_ONLY_FOR_FAMILY="true"
    )
    assert result.returncode != 0
    assert (
        "test_scope=generic cannot be combined with sanity_check_only_for_family"
        in result.stderr
    )
    assert not output.exists()


@pytest.mark.parametrize(
    "skip_stages", ["debug-tools,storage-libs", "debug-tools, storage-libs"]
)
def test_generic_workflow_trims_skipped_stage_names(
    tmp_path: Path, skip_stages: str
) -> None:
    output = tmp_path / "output"
    result = _run_workflow_configure_step(
        SOURCE_ROOT,
        output,
        "generic",
        PROJECTS_TO_TEST="rocgdb-cpu,hipfile",
        SKIP_STAGES=skip_stages,
    )
    assert result.returncode == 0, result.stderr
    outputs = dict(
        line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines()
    )
    assert json.loads(outputs["components"]) == []


@pytest.mark.parametrize(
    "project, expected, build_variant",
    [
        (
            "rocgdb-cpu",
            {
                "rocgdb_run",
                "rocgdb_test",
                "amd-llvm_dev",
                "amd-dbgapi_lib",
                "sysdeps-gmp_lib",
            },
            "release",
        ),
        (
            "hipfile",
            {
                "hipfile_lib",
                "hipfile_test",
                "amd-llvm_run",
                "core-hip_lib",
                "sysdeps-util-linux_lib",
            },
            "asan",
        ),
    ],
)
def test_generic_workflow_passes_effective_artifact_family_to_installer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    project: str,
    expected: set[str],
    build_variant: str,
) -> None:
    jobs = load_workflow(WORKFLOWS_DIR / "test_artifacts.yml")["jobs"]
    assert jobs["configure_test_matrix"]["outputs"]["artifact_family"] == (
        "${{ steps.configure.outputs.artifact_family }}"
    )
    assert jobs["test_components"]["with"]["amdgpu_families"] == (
        "${{ needs.configure_test_matrix.outputs.artifact_family || inputs.amdgpu_families }}"
    )
    assert (
        jobs["test_components"]["with"]["amdgpu_targets"]
        == "${{ inputs.amdgpu_targets }}"
    )
    assert (
        jobs["test_components"]["with"]["artifact_group"]
        == "${{ inputs.artifact_group }}"
    )
    test = load_workflow(WORKFLOWS_DIR / "test_component.yml")["jobs"]["test_component"]
    setup = next(
        step
        for step in test["steps"]
        if step.get("uses") == "./.github/actions/setup_test_environment"
    )
    assert setup["with"]["ARTIFACT_GROUP"] == "${{ inputs.amdgpu_families }}"
    assert setup["with"]["AMDGPU_TARGETS"] == "${{ inputs.amdgpu_targets }}"
    action = load_workflow(
        SOURCE_ROOT / ".github/actions/setup_test_environment/action.yml"
    )
    download = next(
        step
        for step in action["runs"]["steps"]
        if step["name"] == "Download and Unpack Artifacts"
    )
    assert download["env"]["ARTIFACT_GROUP"] == "${{ inputs.ARTIFACT_GROUP }}"
    assert download["env"]["AMDGPU_TARGETS"] == "${{ inputs.AMDGPU_TARGETS }}"
    assert "--artifact-group=${ARTIFACT_GROUP}" in download["run"]
    assert "--amdgpu-targets=${AMDGPU_TARGETS}" in download["run"]

    output_path = tmp_path / "output"
    result = _run_workflow_configure_step(
        SOURCE_ROOT,
        output_path,
        "generic",
        PROJECTS_TO_TEST=project,
        BUILD_VARIANT=build_variant,
    )
    assert result.returncode == 0, result.stderr
    outputs = dict(line.split("=", 1) for line in output_path.read_text().splitlines())
    assert outputs["artifact_family"] == "generic"
    assert json.loads(outputs["sanity_component"]) is None
    (component,) = json.loads(outputs["components"])
    assert component["job_name"] == project

    expected_archives = {f"{artifact}_generic.tar.zst" for artifact in expected}
    available = (
        expected_archives
        | {
            f"{artifact}_{target}.tar.zst"
            for artifact in expected
            for target in ("gfx94X-dcgpu", "gfx942")
        }
        | {"blas_test_generic.tar.zst"}
    )
    monkeypatch.setattr(
        S3Backend, "list_artifacts", lambda self, *args, **kwargs: available
    )
    root = WorkflowOutputRoot.for_local(run_id="12345", platform="linux")
    monkeypatch.setattr(WorkflowOutputRoot, "from_workflow_run", lambda **kwargs: root)
    installer.main(
        [
            "--run-id=12345",
            f"--artifact-group={outputs['artifact_family']}",
            "--amdgpu-targets=",
            *shlex.split(component["fetch_artifact_args"]),
            "--dry-run",
        ]
    )
    stdout = capsys.readouterr().out
    assert {
        name for name in available if f"/12345-linux/{name}" in stdout
    } == expected_archives
