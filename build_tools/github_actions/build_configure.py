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
build_dir = os.getenv("build_dir")
vctools_install_dir = os.getenv("vctools_install_dir")
github_workspace = os.getenv("GITHUB_WORKSPACE")

platform_options = {
    "linux": [
        "-DTHEROCK_VERBOSE=ON",
        "-DBUILD_TESTING=ON"
    ],
    "windows": [
        f"-DCMAKE_C_COMPILER={vctools_install_dir}/bin/Hostx64/x64/cl.exe"
        f"-DCMAKE_CXX_COMPILER={vctools_install_dir}/bin/Hostx64/x64/cl.exe",
        f"-DCMAKE_LINKER={vctools_install_dir}/bin/Hostx64/x64/link.exe",
        "-DTHEROCK_BACKGROUND_BUILD_JOBS=4",
        "-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded",
        f"-DTHEROCK_AMDGPU_WINDOWS_INTEROP_DIR={github_workspace}/amdgpu-windows-interop"
    ]
}


def build_configure():
    logging.info(f"Building package {package_version}")

    cmd = [
        "cmake",
        "-B",
        build_dir,
        "-GNinja",
        ".",
        f"-DTHEROCK_AMDGPU_FAMILIES={amdgpu_families}",
        f"-DTHEROCK_PACKAGE_VERSION='{package_version}'",
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
    ]
    
    # Adding platform specific options
    cmd += platform_options[PLATFORM]
        
    # Splitting cmake options into an array (ex: "-flag X" -> ["-flag", "X"]) for subprocess.run
    cmake_options_arr = extra_cmake_options.split()
    cmd += cmake_options_arr
    
    logging.info(shlex.join(cmd))
    subprocess.run(cmd, cwd=THEROCK_DIR, check=True)


if __name__ == "__main__":
    build_configure()
