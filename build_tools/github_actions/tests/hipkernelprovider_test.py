# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

WRAPPER = (
    Path(__file__).resolve().parent.parent
    / "test_executable_scripts"
    / "test_hipkernelprovider.py"
)


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    # Spaces and glob metacharacters must remain literal filesystem paths.
    root = tmp_path / "artifact [build]"
    wheel_dir = root / "hip_kernel_provider" / "wheels"
    wheel_dir.mkdir(parents=True)
    monkeypatch.setenv("THEROCK_BIN_DIR", str(root))
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("ROCKE_COMGR_LIB", raising=False)
    return root


def stage_wheel(artifact: Path, name: str) -> str:
    path = artifact / "hip_kernel_provider" / "wheels" / name
    path.touch()
    return str(path)


COMPILER = json.dumps(
    {"requested_comgr": "selected-compiler", "llvm_version": [23, 0, 1]}
)


def run_wrapper(run):
    with mock.patch("subprocess.run", run), pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(WRAPPER), run_name="__main__")
    return exit_info.value.code


def test_platform_extra_resolves_sdk_without_reinstalling_dependencies(artifact):
    platform = stage_wheel(artifact, "rocke-0.1-cp312-cp312-linux_x86_64.whl")
    library = stage_wheel(artifact, "rocke_library-0.1-py3-none-any.whl")
    run = mock.Mock(return_value=subprocess.CompletedProcess([], 0, stdout=COMPILER))

    assert run_wrapper(run) == 0

    platform_call, library_call, probe_call, runner_call = run.call_args_list
    command = platform_call.args[0]
    assert command[:3] == ["uv", "pip", "install"]
    assert command[-1] == platform + "[sdk-core]"
    assert "--no-deps" not in command
    assert "--reinstall" not in command
    assert command[command.index("--reinstall-package") + 1] == "rocke"
    assert command[command.index("--index") + 1] == (
        "https://nightly.repo.amd.com/rocm/whl-next/"
    )
    assert command[command.index("--default-index") + 1] == "https://pypi.org/simple/"
    assert command[command.index("--prerelease") + 1] == "allow"
    assert command[command.index("--python") + 1] == sys.executable
    assert platform_call.kwargs == {"check": True}
    assert library_call.args[0] == [
        "uv",
        "pip",
        "install",
        "--no-deps",
        "--reinstall",
        "--python",
        sys.executable,
        library,
    ]
    assert library_call.kwargs == {"check": True}
    assert probe_call.args[0][:2] == [sys.executable, "-c"]
    assert probe_call.kwargs == {"check": True, "capture_output": True, "text": True}
    assert os.environ["ROCKE_COMGR_LIB"] == "selected-compiler"
    assert runner_call.args[0] == [
        sys.executable,
        str(WRAPPER.with_name("test_runner.py")),
    ]


def test_library_only_preserves_dependency_free_install(artifact):
    library = stage_wheel(artifact, "rocke_library-0.1-py3-none-any.whl")
    run = mock.Mock(return_value=subprocess.CompletedProcess([], 0, stdout=COMPILER))

    assert run_wrapper(run) == 0
    assert len(run.call_args_list) == 2
    assert run.call_args_list[0].args[0] == [
        "uv",
        "pip",
        "install",
        "--no-deps",
        "--reinstall",
        "--python",
        sys.executable,
        library,
    ]


@pytest.mark.parametrize("failed_install", [0, 1])
def test_install_failure_stops_before_ctest(artifact, failed_install, capsys):
    stage_wheel(artifact, "rocke-0.1-cp312-cp312-linux_x86_64.whl")
    stage_wheel(artifact, "rocke_library-0.1-py3-none-any.whl")
    results = [subprocess.CompletedProcess([], 0, stdout=COMPILER)] * failed_install
    results.append(subprocess.CalledProcessError(7, "uv"))
    run = mock.Mock(side_effect=results)

    assert run_wrapper(run) == 1
    assert run.call_count == failed_install + 1
    assert "failed with exit code 7" in capsys.readouterr().err


def test_missing_uv_stops_before_ctest(artifact, capsys):
    stage_wheel(artifact, "rocke-0.1-cp312-cp312-linux_x86_64.whl")
    run = mock.Mock(side_effect=FileNotFoundError("uv"))

    assert run_wrapper(run) == 1
    assert run.call_count == 1
    assert "could not run uv" in capsys.readouterr().err


@pytest.mark.parametrize("returncode, expected", [(0, 0), (7, 7), (-15, 143)])
def test_no_wheels_forwards_ctest_status(artifact, returncode, expected):
    run = mock.Mock(return_value=subprocess.CompletedProcess([], returncode))

    assert run_wrapper(run) == expected
    run.assert_called_once_with(
        [sys.executable, str(WRAPPER.with_name("test_runner.py"))]
    )


def test_package_fallback_preserves_pythonpath_and_explicit_comgr(
    artifact, monkeypatch
):
    (artifact / "hip_kernel_provider" / "rocke").mkdir()
    monkeypatch.setenv("PYTHONPATH", "existing-package")
    monkeypatch.setenv("ROCKE_COMGR_LIB", "explicit-comgr")
    expected = os.pathsep.join(
        [str(artifact / "hip_kernel_provider"), "existing-package"]
    )

    def run(command, **kwargs):
        assert os.environ["PYTHONPATH"] == expected
        assert os.environ["ROCKE_COMGR_LIB"] == "explicit-comgr"
        return subprocess.CompletedProcess(command, 0, stdout=COMPILER)

    assert run_wrapper(run) == 0


def test_missing_artifact_environment_fails_before_subprocess(monkeypatch, capsys):
    monkeypatch.delenv("THEROCK_BIN_DIR", raising=False)
    run = mock.Mock()

    assert run_wrapper(run) == 1
    run.assert_not_called()
    assert "THEROCK_BIN_DIR environment variable is required" in capsys.readouterr().err


@pytest.mark.parametrize("override", [None, "", "explicit-compiler"])
def test_selected_compiler_reaches_native_runner(artifact, monkeypatch, override):
    stage_wheel(artifact, "rocke-0.1-py3-none-any.whl")
    if override is not None:
        monkeypatch.setenv("ROCKE_COMGR_LIB", override)
    expected = override or "selected-compiler"
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[-1] == str(WRAPPER.with_name("test_runner.py")):
            assert os.environ["ROCKE_COMGR_LIB"] == expected
        return subprocess.CompletedProcess(command, 0, stdout=COMPILER)

    assert run_wrapper(run) == 0
    assert len(calls) == 3


@pytest.mark.parametrize(
    "failure", ["probe-error", "bad-json", "no-path", "no-version"]
)
def test_compiler_setup_failure_stops_before_ctest(artifact, capsys, failure):
    stage_wheel(artifact, "rocke-0.1-py3-none-any.whl")
    result = {
        "probe-error": subprocess.CalledProcessError(
            1, "probe", stderr="no SDK payload"
        ),
        "bad-json": subprocess.CompletedProcess([], 0, stdout="not json"),
        "no-path": subprocess.CompletedProcess(
            [], 0, stdout=json.dumps({"llvm_version": [23, 0, 1]})
        ),
        "no-version": subprocess.CompletedProcess(
            [],
            0,
            stdout=json.dumps({"requested_comgr": "compiler", "llvm_version": None}),
        ),
    }[failure]
    run = mock.Mock(side_effect=[subprocess.CompletedProcess([], 0), result])
    assert run_wrapper(run) == 1
    assert run.call_count == 2
    assert "compiler" in capsys.readouterr().err
