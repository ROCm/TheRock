# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
This script runs the Linux and Windows build configurations

Required environment variables:
  - amdgpu_families
  - package_version
  - extra_cmake_options
  - BUILD_DIR

Optional environment variables:
  - dist_amdgpu_families: Semicolon- or comma-separated dist family union for
                          THEROCK_DIST_AMDGPU_FAMILIES (TARGET_NEUTRAL device-code
                          libs such as rpp). If unset, gfx94X/gfx950 classic CI
                          shards expand to that pair so they produce the same
                          rpp_lib_generic (ROCm/rocm-libraries#11219).
  - VCToolsInstallDir
  - GITHUB_WORKSPACE
  - EXTRA_C_COMPILER_LAUNCHER: Compiler launcher for C (e.g., resource_info.py for build
                               time analysis). If set, this replaces ccache as the launcher.
                               Note: resource_info.py automatically invokes ccache internally.
  - EXTRA_CXX_COMPILER_LAUNCHER: Compiler launcher for CXX. Same behavior as above.
"""

import argparse
import logging
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess

from manylinux_config import DIST_PYTHON_EXECUTABLES, SHARED_PYTHON_EXECUTABLES

logging.basicConfig(level=logging.INFO)
THIS_SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = THIS_SCRIPT_DIR.parent.parent

PLATFORM = platform.system().lower()

cmake_preset = os.getenv("cmake_preset")
amdgpu_families = os.getenv("amdgpu_families")
dist_amdgpu_families = os.getenv("dist_amdgpu_families")
package_version = os.getenv("package_version")
extra_cmake_options = os.getenv("extra_cmake_options")
github_workspace = os.getenv("GITHUB_WORKSPACE")
extra_c_compiler_launcher = os.getenv("EXTRA_C_COMPILER_LAUNCHER", "")
extra_cxx_compiler_launcher = os.getenv("EXTRA_CXX_COMPILER_LAUNCHER", "")

# Classic TheRock CI Nightly (rocm-libraries therock-ci-nightly.yml) builds
# these families as separate jobs that share a run-id and last-writer-wins
# TARGET_NEUTRAL artifacts. Dist-target device libs (rpp) must compile the
# union or gfx942 testers can receive a gfx950-only librpp.so (#11219).
_CLASSIC_CI_GENERIC_DIST_FAMILIES: tuple[str, ...] = (
    "gfx94X-dcgpu",
    "gfx950-dcgpu",
)
_CLASSIC_CI_GENERIC_DIST_FAMILY_ALIASES: dict[str, str] = {
    "gfx94x": "gfx94X-dcgpu",
    "gfx94x-dcgpu": "gfx94X-dcgpu",
    "gfx950": "gfx950-dcgpu",
    "gfx950-dcgpu": "gfx950-dcgpu",
    "gfx950-all": "gfx950-dcgpu",
}


def _split_amdgpu_families(value: str) -> list[str]:
    """Split a comma/semicolon/whitespace family list into non-empty tokens."""
    return [token for token in re.split(r"[,;\s]+", value.strip()) if token]


def canonicalize_classic_ci_family(family: str) -> str:
    """Map a classic-CI family token to its CMake family name when known."""
    stripped = family.strip()
    if not stripped:
        return stripped
    alias = _CLASSIC_CI_GENERIC_DIST_FAMILY_ALIASES.get(stripped.lower())
    return alias if alias is not None else stripped


def resolve_dist_amdgpu_families(
    shard_families: str | None,
    dist_families: str | None = None,
) -> str | None:
    """Return a CMake THEROCK_DIST_AMDGPU_FAMILIES list, or None for the default.

    An explicit dist_families value always wins. Otherwise, if this shard is one
    of the classic-CI families that upload the same _generic artifacts, expand
    DIST to that union so TARGET_NEUTRAL HIP libraries contain every co-built
    arch. Other shards keep CMake's default (DIST = AMDGPU_FAMILIES).
    """
    if dist_families and dist_families.strip():
        families = [
            canonicalize_classic_ci_family(token)
            for token in _split_amdgpu_families(dist_families)
        ]
        return ";".join(families)

    if not shard_families or not shard_families.strip():
        return None

    shards = [
        canonicalize_classic_ci_family(token)
        for token in _split_amdgpu_families(shard_families)
    ]
    pair = set(_CLASSIC_CI_GENERIC_DIST_FAMILIES)
    if not any(shard in pair for shard in shards):
        return None

    ordered = list(_CLASSIC_CI_GENERIC_DIST_FAMILIES)
    for shard in shards:
        if shard not in ordered:
            ordered.append(shard)
    return ";".join(ordered)


# Normalize paths to use forward slashes for CMake compatibility on Windows
if extra_c_compiler_launcher:
    extra_c_compiler_launcher = extra_c_compiler_launcher.replace("\\", "/")
if extra_cxx_compiler_launcher:
    extra_cxx_compiler_launcher = extra_cxx_compiler_launcher.replace("\\", "/")


def build_compiler_launcher(
    extra_launcher: str, default_launcher: str = "ccache"
) -> str:
    """Build compiler launcher string.

    Args:
        extra_launcher: Custom launcher to use (e.g., resource_info.py).
                        If provided, this replaces the default launcher entirely.
                        Note: resource_info.py automatically invokes ccache internally,
                        so no semicolon-separated list is needed.
        default_launcher: Default launcher to use when extra_launcher is not set.

    Returns:
        Launcher string for CMake. If extra_launcher is provided, returns it directly.
        Otherwise returns default_launcher.

    Example:
        build_compiler_launcher("/path/to/resource_info.py", "ccache")
        -> "/path/to/resource_info.py"

        build_compiler_launcher("", "ccache")
        -> "ccache"
    """
    if extra_launcher:
        return extra_launcher
    return default_launcher


platform_options = {
    "windows": [
        "-DTHEROCK_BACKGROUND_BUILD_JOBS=4",
    ],
}


def build_configure(build_dir: str, manylinux: bool = False) -> None:
    logging.info(f"Building package {package_version}")

    cmd = [
        "cmake",
        "-B",
        build_dir,
        "-GNinja",
        ".",
    ]
    if cmake_preset:
        cmd.extend(["--preset", cmake_preset])
    # Build compiler launcher strings (prepend extra launcher if provided)
    c_launcher = build_compiler_launcher(extra_c_compiler_launcher)
    cxx_launcher = build_compiler_launcher(extra_cxx_compiler_launcher)

    cmd.extend(
        [
            f"-DTHEROCK_AMDGPU_FAMILIES={amdgpu_families}",
            f"-DTHEROCK_PACKAGE_VERSION={package_version}",
            f"-DCMAKE_C_COMPILER_LAUNCHER={c_launcher}",
            f"-DCMAKE_CXX_COMPILER_LAUNCHER={cxx_launcher}",
            "-DBUILD_TESTING=ON",
        ]
    )
    resolved_dist = resolve_dist_amdgpu_families(amdgpu_families, dist_amdgpu_families)
    if resolved_dist:
        logging.info(
            "Setting THEROCK_DIST_AMDGPU_FAMILIES=%s (shard=%s)",
            resolved_dist,
            amdgpu_families,
        )
        cmd.append(f"-DTHEROCK_DIST_AMDGPU_FAMILIES={resolved_dist}")

    # Adding platform specific options
    cmd += platform_options.get(PLATFORM, [])

    # Adding manylinux Python executables if --manylinux is set
    if manylinux:
        cmd.append(f"-DTHEROCK_DIST_PYTHON_EXECUTABLES={DIST_PYTHON_EXECUTABLES}")
        cmd.append("-DTHEROCK_ENABLE_SYSDEPS_AMD_MESA=ON")
        cmd.append("-DTHEROCK_ENABLE_ROCDECODE=ON")
        cmd.append("-DTHEROCK_ENABLE_ROCJPEG=ON")
        cmd.append("-DTHEROCK_ENABLE_RPP=ON")

        # Python executables with shared libpython support. This is needed for
        # ROCgdb.
        cmd.append(f"-DTHEROCK_SHARED_PYTHON_EXECUTABLES={SHARED_PYTHON_EXECUTABLES}")

    # Splitting cmake options into an array (ex: "-flag X" -> ["-flag", "X"]) for subprocess.run
    cmake_options_arr = extra_cmake_options.split()
    cmd += cmake_options_arr

    logging.info(shlex.join(cmd))
    subprocess.run(cmd, cwd=THEROCK_DIR, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run build configuration")
    parser.add_argument(
        "--manylinux",
        action="store_true",
        help="Enable manylinux build with multiple Python versions",
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        default=os.getenv("BUILD_DIR", ""),
        help="Directory to use for build files",
    )
    args = parser.parse_args()

    # Support both command-line flag and environment variable
    manylinux = args.manylinux or os.getenv("MANYLINUX") in ["1", "true"]

    build_configure(args.build_dir, manylinux=manylinux)
