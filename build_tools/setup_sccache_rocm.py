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
import os
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


def cuid_launcher_path() -> Path:
    """Absolute path to the per-TU ``-cuid`` injecting launcher (sibling script)."""
    return Path(__file__).parent.resolve() / "cuid_launcher.py"


def resolve_sccache(explicit_path: Path | None, gha: bool) -> Path | None:
    """Locate and validate a usable sccache binary.

    A missing or non-functional sccache is a config problem that would otherwise
    silently degrade the build to "no cache". Under ``--gha`` we surface it as a
    GitHub Actions ``::warning::`` (visible in the run summary) and return None so
    the build proceeds uncached; locally it is a hard error so the developer
    notices immediately. Flip the ``::warning::`` to ``::error::`` + a non-zero
    exit if a broken cache config should fail CI instead.
    """

    def _unusable(message: str) -> None:
        if gha:
            print(
                f"::warning::sccache unusable: {message}. "
                "Building WITHOUT compiler cache."
            )
            return None
        raise RuntimeError(message)

    if explicit_path is not None:
        if not explicit_path.exists():
            return _unusable(f"specified path not found: {explicit_path}")
        sccache_path = explicit_path
    else:
        sccache_path = find_sccache()
        if not sccache_path:
            return _unusable(
                "not found (install: https://github.com/mozilla/sccache#installation; "
                "for CI it ships in the manylinux build image)"
            )

    try:
        subprocess.run(
            [str(sccache_path), "--version"], capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, OSError) as e:
        return _unusable(f"{sccache_path} failed `--version`: {e}")
    return sccache_path


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
    parser.add_argument(
        "--gha",
        action="store_true",
        help=(
            "Write HIP_CLANG_LAUNCHER (+ THEROCK_SCCACHE) to $GITHUB_ENV for use in "
            "subsequent steps. CMAKE_C/CXX_COMPILER_LAUNCHER are intentionally excluded "
            "— setting them globally breaks stages that use custom compiler wrappers "
            "(e.g. profiler-apps). Those launchers are passed via cmake -D args in the "
            "Configure step instead."
        ),
    )
    parser.add_argument(
        "--no-cuid-launcher",
        action="store_true",
        help=(
            "Point HIP_CLANG_LAUNCHER straight at sccache instead of the cuid_launcher.py "
            "wrapper. The wrapper injects a per-TU -cuid so RDC objects don't collide on "
            "__hip_cuid_ under caching (ROCm/TheRock#748); use this to opt out."
        ),
    )
    args = parser.parse_args()

    sccache_path = resolve_sccache(args.sccache_path, args.gha)
    if sccache_path is None:
        # --gha already emitted a ::warning::; proceed without configuring a cache.
        return
    print(f"Using sccache: {sccache_path}")

    env = sccache_build_env(sccache_path, hip_launcher=not args.no_hip_launcher)
    if args.gha:
        # Only HIP_CLANG_LAUNCHER (+ THEROCK_SCCACHE) is written to $GITHUB_ENV;
        # see the --gha help for why the CMAKE launchers are excluded here.
        gha_vars: dict[str, str] = {}
        if "HIP_CLANG_LAUNCHER" in env:
            if args.no_cuid_launcher:
                gha_vars["HIP_CLANG_LAUNCHER"] = str(sccache_path)
            else:
                # Route hipcc's device compiles through the cuid launcher, which
                # injects a per-TU -cuid ahead of sccache (ROCm/TheRock#748). The
                # wrapper locates the real sccache via THEROCK_SCCACHE.
                gha_vars["THEROCK_SCCACHE"] = str(sccache_path)
                gha_vars["HIP_CLANG_LAUNCHER"] = str(cuid_launcher_path())
        github_env = Path(os.environ["GITHUB_ENV"])
        with github_env.open("a") as f:
            for key, value in gha_vars.items():
                f.write(f"{key}={value}\n")
        for key, value in gha_vars.items():
            print(f"Wrote {key}={value} to $GITHUB_ENV")
    else:
        print("Configure a ROCm build with:")
        for key, value in env.items():
            print(f"  export {key}={value}")


if __name__ == "__main__":
    main()
