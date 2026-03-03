from __future__ import annotations

import os
import sys
from pathlib import Path


def _find_platform_root() -> Path:
    """
    Locate the packaged ROCm platform root inside this wheel.

    Phase 3 will populate runtime files into a subdirectory that contains `bin/`.
    This function finds it dynamically to avoid hard-coding the exact platform tag.
    """
    pkg_dir = Path(__file__).resolve().parent

    candidates: list[Path] = []
    for child in pkg_dir.iterdir():
        if child.is_dir() and (child / "bin").is_dir():
            candidates.append(child)

    if len(candidates) == 1:
        return candidates[0]

    for child in pkg_dir.iterdir():
        if not child.is_dir():
            continue
        for grand in child.iterdir():
            if grand.is_dir() and (grand / "bin").is_dir():
                candidates.append(grand)

    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        "Could not locate packaged ROCm profiler binaries. "
        "Expected a directory containing `bin/` under the installed rocm_sdk_profilers package."
    )


def _exe_suffix() -> str:
    return ".exe" if os.name == "nt" else ""


def _exec(relpath: str) -> None:
    root = _find_platform_root()
    full_path = root / (relpath + _exe_suffix())
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
