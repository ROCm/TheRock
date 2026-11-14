#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Locate sccache and configure it for ROCm HIP builds.

HIP device code is compiled by ``hipcc``, which invokes ``clang`` via absolute
paths and therefore bypasses ``CMAKE_C/CXX_COMPILER_LAUNCHER``. To cache those
device compiles we set ``HIP_CLANG_LAUNCHER`` so that hipcc itself runs clang
through sccache::

    CMD = "<HIP_CLANG_LAUNCHER>" "<clang>" <args>

This leaves the real clang binary in place (so compiler-detection probes such as
hipcc's and torchaudio/torchvision's behave normally) and lets sccache cache the
``-x hip --offload-arch`` device passes -- the expensive, per-architecture part
of a multi-arch build.

Validated on ROCm 7.14 (ROCm/TheRock#5471 investigation): a cold build writes
the HIP compiles to the cache and a subsequent warm build serves them at a 100%
HIP cache-hit rate while the produced wheel still contains device code.

Requires hipcc with HIP_CLANG_LAUNCHER support (ROCm 7.13+,
ROCm/llvm-project#1490). See also ROCm/ROCm#2817 and ROCm/TheRock#3760.

History: an earlier version of this module physically replaced clang/clang++
with wrapper scripts (``sccache <clang> "$@"``). That broke hipcc's
compiler-detection probes (sccache returned ``Compiler not supported`` on the
probe, leaving torch CPU-only) and is no longer used; HIP_CLANG_LAUNCHER
supersedes it.

Usage::

    from setup_sccache_rocm import find_sccache, sccache_build_env
    sccache = find_sccache()
    env.update(sccache_build_env(sccache, hip_launcher=True))

Prerequisites:
    sccache must be installed and available in PATH.
    Install: https://github.com/mozilla/sccache#installation
    For CI, sccache is pre-installed in the manylinux build image:
      https://github.com/ROCm/TheRock/tree/main/dockerfiles
"""

import argparse
import platform
import shutil
import subprocess
from pathlib import Path

is_windows = platform.system() == "Windows"


def find_sccache() -> Path | None:
    """Find sccache binary in PATH or common locations."""
    sccache_path = shutil.which("sccache")
    if sccache_path:
        return Path(sccache_path)

    common_paths = [
        Path("/usr/local/bin/sccache"),
        Path("/opt/cache/bin/sccache"),
        Path.home() / ".cargo" / "bin" / "sccache",
    ]
    if is_windows:
        common_paths.extend(
            [
                Path("C:/ProgramData/chocolatey/bin/sccache.exe"),
                Path.home() / ".cargo" / "bin" / "sccache.exe",
            ]
        )

    for path in common_paths:
        if path.exists():
            return path

    return None


def sccache_build_env(sccache_path: Path, hip_launcher: bool = True) -> dict[str, str]:
    """Return env vars that route a ROCm build's compiles through sccache.

    Always sets ``CMAKE_C_COMPILER_LAUNCHER`` / ``CMAKE_CXX_COMPILER_LAUNCHER``
    (host C/C++ compiles driven by CMake).

    When ``hip_launcher`` is True and not on Windows, also sets
    ``HIP_CLANG_LAUNCHER`` so that hipcc routes its clang invocations -- including
    the HIP device passes that bypass the CMake launchers -- through sccache.

    On Windows, HIP_CLANG_LAUNCHER is omitted: the Windows PyTorch build drives
    clang-cl through the CMake launchers directly rather than via hipcc.
    """
    env = {
        "CMAKE_C_COMPILER_LAUNCHER": str(sccache_path),
        "CMAKE_CXX_COMPILER_LAUNCHER": str(sccache_path),
    }
    if hip_launcher and not is_windows:
        env["HIP_CLANG_LAUNCHER"] = str(sccache_path)
    return env


def main():
    parser = argparse.ArgumentParser(
        description="Locate sccache and print the env to configure a ROCm HIP build."
    )
    parser.add_argument(
        "--sccache-path",
        type=Path,
        help="Path to sccache binary (auto-detected if not specified)",
    )
    parser.add_argument(
        "--no-hip-launcher",
        action="store_true",
        help="Omit HIP_CLANG_LAUNCHER (host C/C++ caching only)",
    )
    args = parser.parse_args()

    if args.sccache_path:
        sccache_path = args.sccache_path
        if not sccache_path.exists():
            raise RuntimeError(f"Specified sccache not found: {sccache_path}")
    else:
        sccache_path = find_sccache()
        if not sccache_path:
            raise RuntimeError(
                "sccache not found.\n"
                "Install: https://github.com/mozilla/sccache#installation\n"
                "For CI, sccache is pre-installed in the manylinux build image:\n"
                "  https://github.com/ROCm/TheRock/tree/main/dockerfiles"
            )

    print(f"Using sccache: {sccache_path}")
    try:
        result = subprocess.run(
            [str(sccache_path), "--version"], capture_output=True, text=True, check=True
        )
        print(f"sccache version: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"sccache verification failed: {e}") from e

    env = sccache_build_env(sccache_path, hip_launcher=not args.no_hip_launcher)
    print("Configure a ROCm build with:")
    for key, value in env.items():
        print(f"  export {key}={value}")


if __name__ == "__main__":
    main()
