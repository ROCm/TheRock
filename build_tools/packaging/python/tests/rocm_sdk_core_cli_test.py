# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the rocm_sdk_core console-script trampoline (`_cli.py`).

Regression coverage for ROCm/TheRock#7075: on Windows the trampoline must
launch the target executable so that a space in the install path (e.g. a user
folder like "C:\\Users\\First Last") is quoted and the child receives an intact
argv[0]. `os.spawnv` did not quote its arguments; `subprocess.run` does.

The `rocm_sdk_core` package normally depends on `_dist_info`, which is generated
at build time and absent from the source tree, so `_cli.py` is loaded in
isolation with a stub `_dist_info` injected.
"""

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

_SRC = Path(__file__).parent.parent / "templates" / "rocm-sdk-core" / "src"


def _load_cli():
    """Load rocm_sdk_core._cli with a stubbed _dist_info dependency."""

    class _StubPackage:
        def __init__(self, name):
            self._name = name
            self.pure_py_package_name = name

        def get_py_package_name(self):
            return self._name

    pkg = types.ModuleType("rocm_sdk_core")
    pkg.__path__ = [str(_SRC / "rocm_sdk_core")]
    dist_info = types.ModuleType("rocm_sdk_core._dist_info")
    dist_info.__version__ = "0.0.0"
    dist_info.ALL_PACKAGES = {
        "core": _StubPackage("_rocm_sdk_core"),
        "devel": _StubPackage("_rocm_sdk_devel"),
    }
    sys.modules["rocm_sdk_core"] = pkg
    sys.modules["rocm_sdk_core._dist_info"] = dist_info

    spec = importlib.util.spec_from_file_location(
        "rocm_sdk_core._cli", _SRC / "rocm_sdk_core" / "_cli.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["rocm_sdk_core._cli"] = module
    spec.loader.exec_module(module)
    return module


cli = _load_cli()

# A module path with a space, mimicking install under "C:\Users\First Last".
_SPACE_PATH = Path(r"C:\Users\First Last\site-packages\_rocm_sdk_core")


def test_exec_windows_passes_space_path_as_single_quoted_arg(monkeypatch):
    monkeypatch.setattr(cli, "_get_module_path", lambda expand_devel=True: _SPACE_PATH)
    monkeypatch.setattr(cli, "is_windows", True)
    monkeypatch.setattr(cli, "exe_suffix", ".exe")
    monkeypatch.setattr(cli.sys, "argv", ["amdclang++", "--help"])

    captured = {}

    class _Result:
        returncode = 7

    def _fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return _Result()

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(SystemExit) as excinfo:
        cli._exec("lib/llvm/bin/amdclang++")

    # Exit code is propagated from the child.
    assert excinfo.value.code == 7

    cmd = captured["cmd"]
    expected_exe = str(_SPACE_PATH / "lib/llvm/bin/amdclang++.exe")
    # The exe path is a single list element (spaces intact), plus forwarded args.
    assert cmd == [expected_exe, "--help"]
    assert " " in cmd[0]
    # Regression guard: subprocess quotes the space-containing path on Windows,
    # which os.spawnv did not do (ROCm/TheRock#7075).
    assert f'"{expected_exe}"' in subprocess.list2cmdline(cmd)


def test_exec_non_windows_uses_execv_with_intact_path(monkeypatch):
    monkeypatch.setattr(cli, "_get_module_path", lambda expand_devel=True: _SPACE_PATH)
    monkeypatch.setattr(cli, "is_windows", False)
    monkeypatch.setattr(cli, "exe_suffix", "")
    monkeypatch.setattr(cli.sys, "argv", ["amdclang++", "--help"])

    captured = {}

    def _fake_execv(path, argv):
        captured["path"] = path
        captured["argv"] = argv

    monkeypatch.setattr(cli.os, "execv", _fake_execv)

    cli._exec("lib/llvm/bin/amdclang++")

    expected_exe = str(_SPACE_PATH / "lib/llvm/bin/amdclang++")
    # execv receives the full path (with space) as argv[0], plus forwarded args.
    assert captured["argv"] == [expected_exe, "--help"]
