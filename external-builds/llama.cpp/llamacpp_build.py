#!/usr/bin/env python3

"""
Automates building llama.cpp with ROCm.

This script configures and builds llama.cpp for AMD GPUs using ROCm.
It assumes that the llama.cpp repository has already been prepared using
the `llamacpp_repo.py` script.

Primary usage:
    ./llamacpp_build.py

Arguments:
    --rocm-dir         : Path to ROCm installation (default: /opt/rocm)
    --llama-dir        : Path to llama.cpp source directory (default: ./llama.cpp)
    --build-dir        : Path to build directory (default: ./llama.cpp/build)
    --cmake-build-type : CMake build type (default: Release)
    --gpu-targets      : AMD GPU target(s) (default: gfx1100)
    --jobs                : Number of jobs for building (passed to cmake --build -j)
    --llama-curl       : Enable libcurl support
    --llama-openssl    : Enable OpenSSL support
    --llama-llguidance : Enable LLGuidance support
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
import subprocess


def set_env_for_rocm(rocm_path: Path, amdgpu_targets: str):
    """Set required ROCm-related environment variables."""
    if not rocm_path.exists():
        sys.exit(f"Error: ROCM_PATH '{rocm_path}' does not exist.")

    # TODO(#1658): when this gets merged into main consider refactoring environment setup
    env_vars = {
        "ROCM_PATH": str(rocm_path),
        "HIP_DEVICE_LIB_PATH": f"{rocm_path}/lib/llvm/amdgcn/bitcode",
    }

    # Set HIPCXX to the ROCm clang compiler
    hip_clang = f"{rocm_path}/llvm/bin/clang"
    if not Path(hip_clang).exists():
        sys.exit(f"Error: HIP compiler not found at {hip_clang}")
    env_vars["HIPCXX"] = hip_clang

    os.environ.update(env_vars)

    print("Environment variables set for ROCm:")
    for k, v in env_vars.items():
        print(f"  {k}={v}")


def build_llamacpp(args: argparse.Namespace):
    rocm_dir = args.rocm_dir
    repo_path = args.llama_dir
    build_dir = args.build_dir

    if not repo_path.exists():
        sys.exit(
            f"Error: Repository path '{repo_path}' does not exist. Run the llamacpp_repo.py script first."
        )

    set_env_for_rocm(rocm_dir, args.gpu_targets)
    build_dir.mkdir(parents=True, exist_ok=True)

    cmake_args = [
        "cmake",
        "-S",
        str(repo_path),
        "-B",
        str(build_dir),
        f"-DGPU_TARGETS={args.gpu_targets}",
        f"-DCMAKE_BUILD_TYPE={args.cmake_build_type}",
        # 3rd party libs
        f"-DLLAMA_CURL={args.llama_curl}",
        f"-DLLAMA_OPENSSL={args.llama_openssl}",
        f"-DLLAMA_LLGUIDANCE={args.llama_llguidance}",
        # Enable HIP backend
        "-DHIP_PLATFORM=amd",
        "-DGGML_HIP=ON",
        # Disable rocWMMA fused attention to avoid build issues
        # TODO: re-evaluate this when rocWMMA is introduced in TheRock
        "-DGGML_HIP_ROCWMMA_FATTN=OFF",
        # Fixes the RPATH to find ROCm shared libraries at runtime
        # (libhipblas.so.3)
        f"-DCMAKE_BUILD_RPATH={rocm_dir}/lib",
        f"-DCMAKE_INSTALL_RPATH={rocm_dir}/lib",
    ]

    print("\nCMake flags set:")
    for arg in cmake_args:
        if arg.startswith("-D"):
            print(f"  {arg}")

    print(f"\nRunning CMake configuration for llama.cpp in {build_dir} ...")
    subprocess.run(cmake_args, check=True)

    print(f"\nBuilding llama.cpp in {build_dir} ...")
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            args.cmake_build_type,
            "--",
            f"-j{args.jobs}",
        ],
        check=True,
    )

    print(f"\nllama.cpp built successfully in {build_dir}")


def main(argv):
    if os.name != "posix":
        sys.exit("This script is intended for Unix-like systems.")

    p = argparse.ArgumentParser(description="Build llama.cpp with ROCm support")
    p.add_argument("--rocm-dir", type=Path, default=Path("/opt/rocm"))
    p.add_argument("--llama-dir", type=Path, default=Path("./llama.cpp"))
    p.add_argument("--build-dir", type=Path, default=Path("./llama.cpp/build"))
    p.add_argument("--cmake-build-type", type=str, default="Release")
    p.add_argument("--gpu-targets", type=str, default="gfx1100")
    p.add_argument(
        "--jobs",
        type=str,
        default="",
        help="Number of jobs for building (passed to cmake --build -1)",
    )
    # 3rd party libs
    p.add_argument("--llama-curl", action="store_true", help="Enable libcurl support")
    p.add_argument(
        "--llama-openssl", action="store_true", help="Enable OpenSSL support"
    )
    p.add_argument(
        "--llama-llguidance", action="store_true", help="Enable LLGuidance support"
    )

    args = p.parse_args(argv)

    print("\nArguments received:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    build_llamacpp(args)


if __name__ == "__main__":
    main(sys.argv[1:])
