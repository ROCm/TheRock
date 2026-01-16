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
  - EXTERNAL_SOURCE_CHECKOUT - Whether building for external repo (true/false)
"""

import argparse
import logging
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

# Add parent directories to path to import detect_external_repo_config
sys.path.insert(0, str(Path(__file__).resolve().parent))

from detect_external_repo_config import (
    detect_repo_name,
    get_repo_config,
    resolve_platform_specific_config,
)

logging.basicConfig(level=logging.INFO)
THIS_SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = THIS_SCRIPT_DIR.parent.parent

PLATFORM = platform.system().lower()

cmake_preset = os.getenv("cmake_preset")
amdgpu_families = os.getenv("amdgpu_families")
package_version = os.getenv("package_version")
extra_cmake_options = os.getenv("extra_cmake_options", "")
build_dir = os.getenv("BUILD_DIR")
vctools_install_dir = os.getenv("VCToolsInstallDir")
github_workspace = os.getenv("GITHUB_WORKSPACE")
external_source_checkout = (
    os.getenv("EXTERNAL_SOURCE_CHECKOUT", "false").lower() == "true"
)

platform_options = {
    "windows": [
        f"-DCMAKE_C_COMPILER={vctools_install_dir}/bin/Hostx64/x64/cl.exe",
        f"-DCMAKE_CXX_COMPILER={vctools_install_dir}/bin/Hostx64/x64/cl.exe",
        f"-DCMAKE_LINKER={vctools_install_dir}/bin/Hostx64/x64/link.exe",
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
    cmd.extend(
        [
            f"-DTHEROCK_AMDGPU_FAMILIES={amdgpu_families}",
            f"-DTHEROCK_PACKAGE_VERSION='{package_version}'",
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            "-DBUILD_TESTING=ON",
        ]
    )

    # Adding platform specific options
    cmd += platform_options.get(PLATFORM, [])

    # Adding manylinux Python executables if --manylinux is set
    if manylinux:
        python_executables = (
            "/opt/python/cp38-cp38/bin/python;"
            "/opt/python/cp39-cp39/bin/python;"
            "/opt/python/cp310-cp310/bin/python;"
            "/opt/python/cp311-cp311/bin/python;"
            "/opt/python/cp312-cp312/bin/python;"
            "/opt/python/cp313-cp313/bin/python"
        )
        cmd.append(f"-DTHEROCK_DIST_PYTHON_EXECUTABLES={python_executables}")
        cmd.append("-DTHEROCK_ENABLE_SYSDEPS_AMD_MESA=ON")

    # Handle external source directory override
    if external_source_checkout:
        repo_override = os.getenv(
            "GITHUB_REPOSITORY_OVERRIDE", os.getenv("GITHUB_REPOSITORY", "")
        )
        if repo_override:
            try:
                repo_name = detect_repo_name(repo_override)
                config = get_repo_config(repo_name)
                platform_config = resolve_platform_specific_config(config, PLATFORM)

                # Add the CMake source directory variable
                cmake_source_var = platform_config.get("cmake_source_var")
                submodule_path = platform_config.get("submodule_path")
                if cmake_source_var and submodule_path:
                    cmd.append(f"-D{cmake_source_var}=./{submodule_path}")
                    logging.info(
                        f"External source override: -D{cmake_source_var}=./{submodule_path}"
                    )
            except (ValueError, KeyError) as e:
                logging.warning(
                    f"Could not determine external source configuration: {e}"
                )

    if PLATFORM == "windows":
        # VCToolsInstallDir is required for build. Throwing an error if environment variable doesn't exist
        if not vctools_install_dir:
            raise Exception(
                "Environment variable VCToolsInstallDir is not set. Please see https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md#important-tool-settings about Windows tool configurations. Exiting."
            )

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
