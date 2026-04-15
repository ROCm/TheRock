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
import shlex
import subprocess

from manylinux_config import DIST_PYTHON_EXECUTABLES, SHARED_PYTHON_EXECUTABLES

logging.basicConfig(level=logging.INFO)
THIS_SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = THIS_SCRIPT_DIR.parent.parent

PLATFORM = platform.system().lower()

cmake_preset = os.getenv("cmake_preset")
amdgpu_families = os.getenv("amdgpu_families")
package_version = os.getenv("package_version")
extra_cmake_options = os.getenv("extra_cmake_options")
github_workspace = os.getenv("GITHUB_WORKSPACE")
extra_c_compiler_launcher = os.getenv("EXTRA_C_COMPILER_LAUNCHER", "")
extra_cxx_compiler_launcher = os.getenv("EXTRA_CXX_COMPILER_LAUNCHER", "")

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

# Map from individual GPU target tokens to required host CPU march flags.
#
# Only list targets here when ALL hardware that ships with that GPU also uses
# a known, fixed CPU microarchitecture (i.e. unified/integrated chipsets where
# the GPU and CPU die are the same product line).
#
# To add a new target: find it in cmake/therock_amdgpu_targets.cmake, verify
# the CPU arch, then add the gfx token → march string entry below.
#
# IMPORTANT: Do NOT add family aliases (e.g. "gfx115X-igpu") unless every
# member of that family uses the same CPU arch.  The gfx115X-igpu family is a
# concrete example of why: it includes gfx1153 (Radeon 820M / Hawk Point,
# Zen 4), so the family cannot be mapped to znver5.
AMDGPU_HOST_MARCH_MAP = {
    # gfx115X Strix family — all ship on Zen 5 CPU dies (Ryzen AI 300 / MAX)
    "gfx1150": "znver5",  # Strix Point  — Ryzen AI 9 HX 370 / AI 9 365
    "gfx1151": "znver5",  # Strix Halo   — Ryzen AI MAX 395 / 385 / 370
    "gfx1152": "znver5",  # Krackan Point — Zen 5c, Ryzen AI 300 mobile
    # gfx1153 (Radeon 820M / Hawk Point) is Zen 4 — intentionally omitted.
}


def host_march_for_families(families_str: str | None) -> str | None:
    """Return a -march value if all requested GPU families map to a single
    known host microarchitecture, otherwise return None."""
    if not families_str:
        return None
    tokens = [t.strip() for t in families_str.replace(";", ",").split(",")]
    known_tokens = [t for t in tokens if t in AMDGPU_HOST_MARCH_MAP]
    marches = {AMDGPU_HOST_MARCH_MAP[t] for t in known_tokens}
    # Only inject if every token maps to the same arch (avoid mixing targets).
    if known_tokens and len(marches) == 1 and len(known_tokens) == len(tokens):
        return marches.pop()
    return None


def build_configure(build_dir, manylinux=False):
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

    # Inject host CPU march flags for targets tied to a specific microarchitecture.
    # MSVC does not accept -march, so skip on Windows.
    if PLATFORM != "windows":
        host_march = host_march_for_families(amdgpu_families)
        if host_march:
            march_flags = f"-march={host_march} -mtune={host_march}"
            cmd.append(f"-DCMAKE_C_FLAGS={march_flags}")
            cmd.append(f"-DCMAKE_CXX_FLAGS={march_flags}")
            logging.info(f"Injecting host CPU flags: {march_flags}")

    # Adding platform specific options
    cmd += platform_options.get(PLATFORM, [])

    # Adding manylinux Python executables if --manylinux is set
    if manylinux:
        cmd.append(f"-DTHEROCK_DIST_PYTHON_EXECUTABLES={DIST_PYTHON_EXECUTABLES}")
        cmd.append("-DTHEROCK_ENABLE_SYSDEPS_AMD_MESA=ON")
        cmd.append("-DTHEROCK_ENABLE_ROCDECODE=ON")
        cmd.append("-DTHEROCK_ENABLE_ROCJPEG=ON")

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
