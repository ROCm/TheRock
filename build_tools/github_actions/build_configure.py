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

logging.basicConfig(level=logging.INFO)
THIS_SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = THIS_SCRIPT_DIR.parent.parent

PLATFORM = platform.system().lower()

cmake_preset = os.getenv("cmake_preset")
amdgpu_families = os.getenv("amdgpu_families")
package_version = os.getenv("package_version")
extra_cmake_options = os.getenv("extra_cmake_options")
build_dir = os.getenv("BUILD_DIR")
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


def build_configure(manylinux=False):
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
            f"-DTHEROCK_PACKAGE_VERSION='{package_version}'",
            f"-DCMAKE_C_COMPILER_LAUNCHER={c_launcher}",
            f"-DCMAKE_CXX_COMPILER_LAUNCHER={cxx_launcher}",
            "-DBUILD_TESTING=ON",
        ]
    )

    # Adding platform specific options
    cmd += platform_options.get(PLATFORM, [])

    # Adding manylinux Python executables if --manylinux is set
    if manylinux:
        python_executables = (
            "/opt/python/cp310-cp310/bin/python;"
            "/opt/python/cp311-cp311/bin/python;"
            "/opt/python/cp312-cp312/bin/python;"
            "/opt/python/cp313-cp313/bin/python"
        )
        cmd.append(f"-DTHEROCK_DIST_PYTHON_EXECUTABLES={python_executables}")
        cmd.append("-DTHEROCK_ENABLE_SYSDEPS_AMD_MESA=ON")

        # Python executables with shared libpython support. This is needed for
        # ROCgdb.
        python_shared_executables = (
            "/opt/python-shared/cp310-cp310/bin/python3;"
            "/opt/python-shared/cp311-cp311/bin/python3;"
            "/opt/python-shared/cp312-cp312/bin/python3;"
            "/opt/python-shared/cp313-cp313/bin/python3;"
            "/opt/python-shared/cp314-cp314/bin/python3"
        )
        cmd.append(f"-DTHEROCK_SHARED_PYTHON_EXECUTABLES={python_shared_executables}")

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
    args = parser.parse_args()

    # Support both command-line flag and environment variable
    manylinux = args.manylinux or os.getenv("MANYLINUX") in ["1", "true"]

    build_configure(manylinux=manylinux)
