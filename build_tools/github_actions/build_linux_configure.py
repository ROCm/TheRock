"""
This script runs the Linux build configuration

Required environment variables:
  - amdgpu_families
  - package_version
  - extra_cmake_options
"""

import logging
import os
from pathlib import Path
import shlex
import subprocess

logging.basicConfig(level=logging.INFO)
THIS_SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = THIS_SCRIPT_DIR.parent.parent

amdgpu_families = os.getenv("amdgpu_families")
package_version = os.getenv("package_version")
extra_cmake_options = os.getenv("extra_cmake_options")


def build_linux_configure():
    logging.info(f"Building package {package_version}")

    # Splitting cmake options into an array (ex: "-flag X" -> ["-flag", "X"]) for subprocess.run
    cmake_options_arr = extra_cmake_options.split()

    cmd = [
        "cmake",
        "-B",
        "build",
        "-GNinja",
        ".",
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        f"-DTHEROCK_AMDGPU_FAMILIES={amdgpu_families}",
        f"-DTHEROCK_PACKAGE_VERSION='{package_version}'",
        "-DTHEROCK_VERBOSE=ON",
        "-DBUILD_TESTING=ON",
    ] + cmake_options_arr
    logging.info(shlex.join(cmd))
    subprocess.run(cmd, cwd=THEROCK_DIR, check=True)


if __name__ == "__main__":
    build_linux_configure()
