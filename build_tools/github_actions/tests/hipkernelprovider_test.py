# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

WRAPPER = (
    Path(__file__).resolve().parent.parent
    / "test_executable_scripts"
    / "test_hipkernelprovider.py"
)
RUNNER = str(WRAPPER.with_name("test_runner.py"))


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    # Restore wrapper mutations too, including variables originally absent.
    monkeypatch.setenv("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv("ROCKE_COMGR_LIB", os.environ.get("ROCKE_COMGR_LIB", ""))
    root = tmp_path / "artifact [build]"
    (root / "hip_kernel_provider" / "wheels").mkdir(parents=True)
    monkeypatch.setenv("THEROCK_BIN_DIR", str(root))
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("ROCKE_COMGR_LIB", raising=False)
    monkeypatch.setenv("ROCKE_HIP_LIB", "caller-artifact-hip")
    return root


def stage_wheel(artifact: Path, name: str) -> str:
    path = artifact / "hip_kernel_provider" / "wheels" / name
    path.touch()
    return str(path)


def stage_compiler(artifact: Path, info: dict | None = None) -> str:
    """Exercise the probe as a real child, with a controlled compiler adapter."""
    runtime = artifact / "hip_kernel_provider" / "rocke" / "runtime"
    runtime.mkdir(parents=True)
    (runtime.parent / "__init__.py").touch()
    (runtime / "__init__.py").touch()
    compiler = artifact / "compiler module"
    compiler.touch()
    values = dict(
        llvm_version=[23, 0, 1],
        source="LLVMGetVersion",
        requested_comgr=str(compiler),
        comgr_path=str(compiler),
        query_library_path="compiler-query-module",
    )
    values.update(info or {})
    (runtime / "comgr.py").write_text(
        "import os\n"
        "from dataclasses import make_dataclass\n"
        f"values = {values!r}\n"
        "def _resolve_lib():\n"
        "    if os.getenv('PROBE_FAIL'):\n"
        "        raise RuntimeError('unloadable compiler')\n"
        "def loaded_compiler_info():\n"
        "    return make_dataclass('CompilerInfo', values)(**values)\n",
        encoding="utf-8",
    )
    return str(compiler)


def run_wrapper(monkeypatch, calls: list, status: int = 0) -> int:
    real_run = subprocess.run

    def run(command, **kwargs):
        calls.append((command, os.environ.copy()))
        if command[:2] == [sys.executable, "-c"]:
            return real_run(command, **kwargs)
        return subprocess.CompletedProcess(
            command, status if command[-1] == RUNNER else 0
        )

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(WRAPPER), run_name="__main__")
    return result.value.code


@pytest.mark.parametrize("override", [None, "", "explicit"])
def test_compiler_and_provenance_reach_ctest(artifact, monkeypatch, caplog, override):
    platform = stage_wheel(artifact, "rocke-0.1.0-py3-none-any.whl")
    library = stage_wheel(artifact, "rocke_library-0.1.0-py3-none-any.whl")
    selected = stage_compiler(artifact)
    if override == "explicit":
        override = selected
    if override is not None:
        monkeypatch.setenv("ROCKE_COMGR_LIB", override)
    calls = []
    with caplog.at_level("INFO"):
        assert run_wrapper(monkeypatch, calls) == 0
    install, probe, runner = calls
    assert install[0] == [
        "uv",
        "pip",
        "install",
        "--no-deps",
        "--reinstall",
        "--python",
        sys.executable,
        platform,
        library,
    ]
    assert probe[0][:2] == [sys.executable, "-c"]
    assert runner[0] == [sys.executable, RUNNER]
    assert runner[1]["ROCKE_COMGR_LIB"] == selected
    assert runner[1]["ROCKE_HIP_LIB"] == "caller-artifact-hip"
    assert "compiler-query-module" in caplog.text
    assert sys.executable in caplog.text


@pytest.mark.parametrize("status, expected", [(0, 0), (7, 7), (-15, 143)])
def test_native_only_artifact_preserves_ctest_status(
    artifact, monkeypatch, status, expected
):
    calls = []
    assert run_wrapper(monkeypatch, calls, status) == expected
    assert [command for command, _ in calls] == [[sys.executable, RUNNER]]


@pytest.mark.parametrize(
    "info, error",
    [
        ({"llvm_version": None}, "could not be loaded and queried"),
        ({"comgr_path": None}, "physical module path"),
        ({"comgr_path": "missing-library"}, "physical module path"),
        ({"requested_comgr": "fallback-compiler"}, "requested ROCKE_COMGR_LIB"),
    ],
)
def test_failed_probe_stops_before_ctest(artifact, monkeypatch, capsys, info, error):
    stage_compiler(artifact, info)
    if "requested_comgr" in info:
        monkeypatch.setenv("ROCKE_COMGR_LIB", "broken-override")
    calls = []
    assert run_wrapper(monkeypatch, calls) == 1
    assert len(calls) == 1
    assert error in capsys.readouterr().err


def test_unloadable_override_stops_before_ctest(artifact, monkeypatch, capsys):
    stage_compiler(artifact)
    monkeypatch.setenv("PROBE_FAIL", "1")
    monkeypatch.setenv("ROCKE_COMGR_LIB", "unloadable-override")
    calls = []
    assert run_wrapper(monkeypatch, calls) == 1
    assert len(calls) == 1
    assert "unloadable compiler" in capsys.readouterr().err


def test_package_fallback_preserves_pythonpath(artifact, monkeypatch):
    selected = stage_compiler(artifact)
    monkeypatch.setenv("PYTHONPATH", "existing-package")
    calls = []
    assert run_wrapper(monkeypatch, calls) == 0
    assert calls[-1][1]["PYTHONPATH"] == os.pathsep.join(
        [str(artifact / "hip_kernel_provider"), "existing-package"]
    )
    assert calls[-1][1]["ROCKE_COMGR_LIB"] == selected


@pytest.mark.parametrize(
    "failure", [FileNotFoundError("uv"), subprocess.CalledProcessError(7, "uv")]
)
def test_wheel_install_failure_stops_before_probe(artifact, monkeypatch, failure):
    stage_wheel(artifact, "rocke-0.1.0-py3-none-any.whl")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        raise failure

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(WRAPPER), run_name="__main__")
    assert result.value.code == 1
    assert len(calls) == 1
