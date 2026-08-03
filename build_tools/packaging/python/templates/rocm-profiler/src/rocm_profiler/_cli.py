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


def _extend_ld_library_path(env: dict[str, str]) -> None:
    """Set up LD_LIBRARY_PATH before we exec into the native rocprof-sys binary.

    These pip entry points replace the current process, so we can't use the
    preload path in python_packaging.md paths go in the env passed to execve.
    librocprof-sys-rt.so lives under site-packages/lib; the loader won't find it
    there on its own, and instrumented binaries need it too. sdk-core's _cli.py
    doesn't do this because those wrappers don't ship runtime .so's the same way.
    """
    if platform.system() == "Windows":
        return

    paths_to_add: list[Path] = []

    profiler_lib_path = _get_profiler_module_path() / "lib"
    if profiler_lib_path.exists():
        paths_to_add.append(profiler_lib_path)

    try:
        sdk_module = importlib.import_module("_rocm_sdk_core")
    except ModuleNotFoundError:
        sdk_module = None

    if sdk_module is not None:
        sdk_path = Path(sdk_module.__file__).parent
        sysdeps_path = sdk_path / "lib" / "rocm_sysdeps" / "lib"
        if sysdeps_path.exists():
            paths_to_add.append(sysdeps_path)

    if not paths_to_add:
        return

    existing = env.get("LD_LIBRARY_PATH", "")
    existing_parts = [p for p in existing.split(":") if p]

    new_parts: list[str] = []
    for path in paths_to_add:
        path_str = str(path)
        if path_str not in existing_parts and path_str not in new_parts:
            new_parts.append(path_str)

    if new_parts:
        env["LD_LIBRARY_PATH"] = ":".join([*new_parts, *existing_parts])


def _extend_pythonpath_for_compute(env: dict[str, str]) -> None:
    """Set up PYTHONPATH for rocprofiler-compute before execve.

    rocprof-sys shells out to Python helpers under libexec/rocprofiler-compute,
    and child processes inherit whatever env we hand off. sdk-core's _cli.py
    has nothing like this its native tools don't pull in a libexec tree.
    """
    if platform.system() == "Windows":
        return

    profiler_root = _get_profiler_module_path()
    compute_path = profiler_root / "libexec" / "rocprofiler-compute"

    if not compute_path.exists():
        return

    existing = env.get("PYTHONPATH", "")
    parts = [p for p in existing.split(":") if p]

    compute_str = str(compute_path)

    if compute_str not in parts:
        env["PYTHONPATH"] = ":".join([compute_str, *parts])


def _get_profiler_module_path() -> Path:
    profiler_module = importlib.import_module(PROFILER_PY_PACKAGE_NAME)
    return Path(profiler_module.__file__).parent


def _exec(relpath: str) -> None:
    if platform.system() == "Windows":
        raise RuntimeError("rocm-profiler is not supported on Windows.")

    # Copy env and execve so path tweaks don't stick around in os.environ.
    env = os.environ.copy()
    _extend_ld_library_path(env)
    _extend_pythonpath_for_compute(env)

    full_path = _get_profiler_module_path() / relpath
    if not full_path.exists():
        raise FileNotFoundError(f"Profiler tool not found: {full_path}")
    os.execve(str(full_path), [str(full_path)] + sys.argv[1:], env)


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
