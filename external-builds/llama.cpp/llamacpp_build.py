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
    --jobs             : Number of jobs for building (passed to cmake --build -j)
    --llama-curl       : Enable libcurl support
    --llama-openssl    : Enable OpenSSL support
    --llama-llguidance : Enable LLGuidance support
    --clean            : Clean build directory before building
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
import subprocess


def setup_environment(rocm_path: Path, amdgpu_targets: str):
    """Set required ROCm-related environment variables."""
    if not rocm_path.exists():
        sys.exit(f"Error: ROCM_PATH '{rocm_path}' does not exist.")

    # TODO(#1658): when this gets merged into main consider refactoring environment setup
    env = {
        "ROCM_PATH": str(rocm_path),
    }

    # Workaround missing devicelib bitcode. Same comment as in
    # /external-builds/pytorch/build_prod_wheels.py applies.
    # This is bad/annoying. Unfortunately, it has been hardcoded for a long time.
    # So we use a clang env var to force a specific device lib path to workaround
    # the hack to get pytorch to build. This may or may not only affect the
    # Python wheels with their own quirks on directory layout.
    # Obviously, this should be completely burned with fire once the root causes
    # are eliminted.
    hip_device_lib_path = rocm_path / "lib/llvm/amdgcn/bitcode"
    if not hip_device_lib_path.exists():
        print(
            "WARNING: Default location of device libs not found. Relying on "
            "clang heuristics which are known to be buggy in this configuration"
        )
    else:
        env["HIP_DEVICE_LIB_PATH"] = str(hip_device_lib_path)

    hip_clang = f"{rocm_path}/llvm/bin/clang"
    if not Path(hip_clang).exists():
        sys.exit(f"Error: HIP compiler not found at {hip_clang}")
    env["HIPCXX"] = hip_clang

    os.environ.update(env)

    print("Environment variables set for ROCm:")
    for k, v in env.items():
        print(f"  {k}={v}")


def build(args: argparse.Namespace):
    rocm_dir = args.rocm_dir
    repo_path = args.llama_dir
    build_dir = args.build_dir

    if not repo_path.exists():
        sys.exit(
            f"Error: Repository path '{repo_path}' does not exist. Run the llamacpp_repo.py script first."
        )

    setup_environment(rocm_dir, args.gpu_targets)

    if args.clean and build_dir.exists():
        print(f"Cleaning build directory {build_dir} ...")
        shutil.rmtree(build_dir)

    build_dir.mkdir(parents=True, exist_ok=True)

    gpu_list = [t.strip() for t in args.gpu_targets.split(",")]
    gpu_targets_str = ";".join(gpu_list)  # CMake expects ;-separated

    cmake_args = [
        "cmake",
        "-S",
        str(repo_path),
        "-B",
        str(build_dir),
        f"-DGPU_TARGETS={gpu_targets_str}",
        # Used to specify AMD GPU targets for older llama.cpp
        # repos that use GGML_AMDGPU_TARGETS instead of GPU_TARGETS
        f"-DAMDGPU_TARGETS={gpu_targets_str}",
        f"-DCMAKE_BUILD_TYPE={args.cmake_build_type}",
        f"-DLLAMA_CURL={args.llama_curl}",
        f"-DLLAMA_OPENSSL={args.llama_openssl}",
        f"-DLLAMA_LLGUIDANCE={args.llama_llguidance}",
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
    p.add_argument(
        "--gpu-targets", type=str, default="gfx1100", help="Comma-separated GPU targets"
    )
    p.add_argument(
        "--jobs",
        type=str,
        default="",
        help="Number of jobs for building (passed to cmake --build -j)",
    )
    p.add_argument("--llama-curl", action="store_true", help="Enable libcurl support")
    p.add_argument(
        "--llama-openssl", action="store_true", help="Enable OpenSSL support"
    )
    p.add_argument(
        "--llama-llguidance", action="store_true", help="Enable LLGuidance support"
    )
    p.add_argument(
        "--clean", action="store_true", help="Clean build directory before building"
    )

    args = p.parse_args(argv)

    print("\nArguments received:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    build(args)


if __name__ == "__main__":
    main(sys.argv[1:])
