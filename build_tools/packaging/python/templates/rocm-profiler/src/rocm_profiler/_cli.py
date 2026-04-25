# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Trampoline for console scripts."""


from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
import platform

from ._dist_info import ALL_PACKAGES


PROFILER_PACKAGE = ALL_PACKAGES["profiler"]
PROFILER_PY_PACKAGE_NAME = PROFILER_PACKAGE.get_py_package_name()


def _extend_ld_library_path() -> None:
    if platform.system() == "Windows":
        return

    try:
        sdk_module = importlib.import_module("_rocm_sdk_core")
    except ModuleNotFoundError:
        return

    sdk_path = Path(sdk_module.__file__).parent
    sysdeps_path = sdk_path / "lib" / "rocm_sysdeps" / "lib"
    if not sysdeps_path.exists():
        return

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    existing_parts = [p for p in existing.split(":") if p]
    sysdeps_str = str(sysdeps_path)

    if sysdeps_str not in existing_parts:
        os.environ["LD_LIBRARY_PATH"] = ":".join([sysdeps_str, *existing_parts])


def _extend_pythonpath_for_compute() -> None:
    if platform.system() == "Windows":
        return

    profiler_root = _get_profiler_module_path()
    compute_path = profiler_root / "libexec" / "rocprofiler-compute"

    if not compute_path.exists():
        return

    existing = os.environ.get("PYTHONPATH", "")
    parts = [p for p in existing.split(":") if p]

    compute_str = str(compute_path)

    if compute_str not in parts:
        os.environ["PYTHONPATH"] = ":".join([compute_str, *parts])


def _get_profiler_module_path() -> Path:
    profiler_module = importlib.import_module(PROFILER_PY_PACKAGE_NAME)
    return Path(profiler_module.__file__).parent


def _exec(relpath: str) -> None:
    if platform.system() == "Windows":
        raise RuntimeError("rocm-profiler is not supported on Windows.")

    _extend_ld_library_path()
    _extend_pythonpath_for_compute()

    full_path = _get_profiler_module_path() / relpath
    if not full_path.exists():
        raise FileNotFoundError(f"Profiler tool not found: {full_path}")
    os.execv(str(full_path), [str(full_path)] + sys.argv[1:])


def rocprof_compute() -> None:
    _exec("bin/rocprof-compute")


def rocprof_sys_avail() -> None:
    _exec("bin/rocprof-sys-avail")


def rocprof_sys_causal() -> None:
    _exec("bin/rocprof-sys-causal")


def rocprof_sys_instrument() -> None:
    _exec("bin/rocprof-sys-instrument")


def rocprof_sys_run() -> None:
    _exec("bin/rocprof-sys-run")


def rocprof_sys_sample() -> None:
    _exec("bin/rocprof-sys-sample")


def rocprof_sys_python() -> None:
    _exec("bin/rocprof-sys-python")


def rocprof_sys_attach() -> None:
    _exec("bin/rocprof-sys-attach")
