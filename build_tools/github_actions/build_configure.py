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
"""

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

amdgpu_families = os.getenv("amdgpu_families")
package_version = os.getenv("package_version")
extra_cmake_options = os.getenv("extra_cmake_options")
build_dir = os.getenv("BUILD_DIR")
vctools_install_dir = os.getenv("VCToolsInstallDir")
github_workspace = os.getenv("GITHUB_WORKSPACE")

platform_options = {
    "windows": [
        f"-DCMAKE_C_COMPILER={vctools_install_dir}/bin/Hostx64/x64/cl.exe",
        f"-DCMAKE_CXX_COMPILER={vctools_install_dir}/bin/Hostx64/x64/cl.exe",
        f"-DCMAKE_LINKER={vctools_install_dir}/bin/Hostx64/x64/link.exe",
        "-DTHEROCK_BACKGROUND_BUILD_JOBS=4",
    ],
}

platform_presets = {
    "linux": "linux-release-package",
}


def build_configure():
    preset = platform_presets.get(PLATFORM)
    logging.info(f"Building package {package_version} (preset {preset})")

    cmd = [
        "cmake",
        "-B",
        build_dir,
        "-GNinja",
        ".",
    ]
    if preset:
        cmd.extend(["--preset", str(preset)])
    cmd.extend(
        [
            f"-DTHEROCK_AMDGPU_FAMILIES={amdgpu_families}",
            f"-DTHEROCK_PACKAGE_VERSION='{package_version}'",
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            "-DTHEROCK_VERBOSE=ON",
            "-DBUILD_TESTING=ON",
        ]
    )

    # Adding platform specific options
    cmd += platform_options.get(PLATFORM, [])

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
    build_configure()
