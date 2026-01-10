"""
This script runs the Linux and Windows build configurations

Required environment variables:
  - amdgpu_families

Optional environment variables:
  - package_version (default: "0.0.0")
  - extra_cmake_options (default: "")
  - BUILD_DIR (default: "build")
  - VCToolsInstallDir
  - GITHUB_WORKSPACE
"""

import argparse
import logging
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

# Add parent directory to path to import compute_rocm_package_version
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compute_rocm_package_version import compute_version

logging.basicConfig(level=logging.INFO)
THIS_SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = THIS_SCRIPT_DIR.parent.parent

PLATFORM = platform.system().lower()


def get_platform_options(vctools_install_dir):
    """Get platform-specific CMake options."""
    return {
        "windows": [
            f"-DCMAKE_C_COMPILER={vctools_install_dir}/bin/Hostx64/x64/cl.exe",
            f"-DCMAKE_CXX_COMPILER={vctools_install_dir}/bin/Hostx64/x64/cl.exe",
            f"-DCMAKE_LINKER={vctools_install_dir}/bin/Hostx64/x64/link.exe",
            "-DTHEROCK_BACKGROUND_BUILD_JOBS=4",
        ],
    }


def build_configure(
    amdgpu_families,
    package_version,
    extra_cmake_options,
    build_dir,
    cmake_preset,
    vctools_install_dir,
    manylinux=False,
):
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
    platform_options = get_platform_options(vctools_install_dir)
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
    parser.add_argument(
        "--amdgpu-families",
        default=os.getenv("amdgpu_families", None),
        help="Comma-separated list of AMD GPU families to build for (default: from amdgpu_families env var)",
    )

    # Calculate default package version using compute_version
    default_package_version = os.getenv("package_version", None)
    if not default_package_version:
        try:
            default_package_version = compute_version(release_type="dev")
        except Exception:
            default_package_version = "ADHOCBUILD"

    parser.add_argument(
        "--package-version",
        default=default_package_version,
        help="Package version string (default: computed dev version or from package_version env var)",
    )
    parser.add_argument(
        "--extra-cmake-options",
        default=os.getenv("extra_cmake_options", ""),
        help="Additional CMake options (default: empty or from extra_cmake_options env var)",
    )
    parser.add_argument(
        "--build-dir",
        default=os.getenv("BUILD_DIR", "build"),
        help="Build directory path (default: build or from BUILD_DIR env var)",
    )
    parser.add_argument(
        "--cmake-preset",
        default=os.getenv("cmake_preset", None),
        help="CMake preset to use (default: from cmake_preset env var)",
    )
    parser.add_argument(
        "--vctools-install-dir",
        default=os.getenv("VCToolsInstallDir", None),
        help="Visual C++ tools install directory for Windows builds (default: from VCToolsInstallDir env var)",
    )
    args = parser.parse_args()

    # Validate required arguments
    if not args.amdgpu_families:
        parser.error(
            "Missing required argument: --amdgpu-families\n"
            "Please provide this via command line or set the amdgpu_families environment variable."
        )

    # Support both command-line flag and environment variable
    manylinux = args.manylinux or os.getenv("MANYLINUX") in ["1", "true"]

    build_configure(
        amdgpu_families=args.amdgpu_families,
        package_version=args.package_version,
        extra_cmake_options=args.extra_cmake_options,
        build_dir=args.build_dir,
        cmake_preset=args.cmake_preset,
        vctools_install_dir=args.vctools_install_dir,
        manylinux=manylinux,
    )
