# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Trampoline for console scripts."""


from __future__ import annotations

import os
import sys
from pathlib import Path
import platform


def _find_platform_root() -> Path:
    """
    Locate the packaged ROCm platform root inside this wheel.

    Phase 3 will populate runtime files into a subdirectory that contains `bin/`.
    This function finds it dynamically to avoid hard-coding the exact platform tag.
    """
    pkg_dir = Path(__file__).resolve().parent
    children = list(pkg_dir.iterdir())
    direct_candidates: list[Path] = [
        child for child in children if child.is_dir() and (child / "bin").is_dir()
    ]
    if len(direct_candidates) == 1:
        return direct_candidates[0]
    if len(direct_candidates) > 1:
        raise RuntimeError(
            "Ambiguous packaged ROCm profiler binaries: found multiple direct "
            f"candidates with bin/: {sorted(str(p) for p in direct_candidates)}"
        )

    nested_candidates: list[Path] = []
    for child in children:
        if not child.is_dir():
            continue
        for grand in child.iterdir():
            if grand.is_dir() and (grand / "bin").is_dir():
                nested_candidates.append(grand)

    if len(nested_candidates) == 1:
        return nested_candidates[0]
    if len(nested_candidates) > 1:
        raise RuntimeError(
            "Ambiguous packaged ROCm profiler binaries: found multiple nested "
            f"candidates with bin/: {sorted(str(p) for p in nested_candidates)}"
        )

    raise FileNotFoundError(
        "Could not locate packaged ROCm profiler binaries. "
        "Expected a directory containing `bin/` under the installed "
        "rocm_profiler package."
    )


def _exec(relpath: str) -> None:

    if platform.system() == "Windows":
        raise RuntimeError("rocm-profiler is not supported on Windows.")

    root = _find_platform_root()
    full_path = root / relpath
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
