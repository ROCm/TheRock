#!/usr/bin/env python3

"""
Automates building llama.cpp with ROCm.

This script configures and builds the llama.cpp repository for AMD GPUs using
an installed ROCm SDK. It is intended to be used by developers who already
have a prepared source checkout (see llamacpp_repo.py) and want a reproducible
CMake-based build that integrates TheRock/ROCm toolchain conventions

Assumptions:

- A working CMake installation is available. Ninja is used as the generator by
  default if present (the script passes -G Ninja).
- ROCm SDK is installed and discoverable via rocm_sdk (python -m rocm_sdk).
- The llama.cpp source tree has been prepared by llamacpp_repo.py
  or otherwise exists at the path passed to --llama-dir.

Usage examples:

# Use defaults (discover ROCm root and targets via rocm_sdk)
python3 ./llamacpp_build.py
# Specify ROCm root and number of jobs
python3 ./llamacpp_build.py --rocm-dir /opt/rocm --jobs 16

Arguments:

--rocm-dir            Path to ROCm installation directory (default: rocm_sdk --root)
--llama-dir           Path to llama.cpp source directory (default: ./llama.cpp next to script)
--build-dir           Path to build directory (default: ./llama.cpp/build next to script)
--cmake-build-type    CMake build type (Debug, Release, RelWithDebInfo, MinSizeRel) (default: Release)
--rocm-targets        Comma-separated ROCm targets (e.g., gfx1030,gfx1100). If omitted, discovered via rocm_sdk.
--jobs                Number of jobs for building (passed to cmake --build -j). "0" lets CMake decide (default: 0).
--llama-curl          Enable libcurl support (Boolean flag)
--llama-openssl       Enable OpenSSL support (Boolean flag)
--llama-llguidance    Enable LLGuidance support (Boolean flag)
--use-ccache          Use ccache as the compiler launcher (Boolean flag)
--clean               Clean build directory before building (Boolean flag)

Windows notes
-------------
- The script attempts to locate rc.exe and mt.exe under the Windows Kits
  installation (C:\Program Files (x86)\Windows Kits\10x\bin). When found,
  it automatically sets -DCMAKE_RC_COMPILER and -DCMAKE_MT for CMake.
- Please note that you need to have environemnt variables set up correctly for
  ROCm on Windows, including PATH updates to find ROCm DLLs at runtime.
"""

import os
import sys
import shlex
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Tuple, Optional

script_dir = Path(__file__).resolve().parent


def directory_if_exists(dir: Path) -> Path | None:
    if dir.exists():
        return dir
    else:
        return None


def capture(args: list[str | Path], cwd: Path) -> str:
    args = [str(arg) for arg in args]
    print(f"++ Capture [{cwd}]$ {shlex.join(args)}")
    try:
        return subprocess.check_output(
            args, cwd=str(cwd), stderr=subprocess.STDOUT, text=True
        ).strip()
    except subprocess.CalledProcessError as e:
        print(f"Error capturing output: {e}")
        print(f"Output from the failed command:\n{e.output}")
        return ""


def get_rocm_path(path_name: str) -> Path:
    return Path(
        capture(
            [sys.executable, "-m", "rocm_sdk", "path", f"--{path_name}"], cwd=Path.cwd()
        ).strip()
    )


def get_rocm_sdk_version() -> str:
    return capture(
        [sys.executable, "-m", "rocm_sdk", "version"], cwd=Path.cwd()
    ).strip()


def get_rocm_sdk_targets() -> str:
    # Run `rocm-sdk targets` to get the default architecture
    targets = capture([sys.executable, "-m", "rocm_sdk", "targets"], cwd=Path.cwd())
    if not targets:
        print("Warning: rocm-sdk targets returned empty or failed")
        return ""
    # Convert space-separated targets to comma-separated
    return targets.replace(" ", ",")


def remove_dir_if_exists(dir: Path):
    if dir.exists():
        print(f"++ Removing {dir} ...")
        shutil.rmtree(dir)


def find_windows_kits_tools() -> Tuple[Optional[Path], Optional[Path]]:
    """
    Find rc.exe and mt.exe under Windows Kits 10 bin directory.
    Returns (rc_path, mt_path) or (None, None) if not found.
    """
    base = Path(r"C:\Program Files (x86)\Windows Kits\10\bin")
    # Try direct existence
    if not base.exists():
        # fallback to PATH lookup
        rc = Path(shutil.which("rc.exe")) if shutil.which("rc.exe") else None
        mt = Path(shutil.which("mt.exe")) if shutil.which("mt.exe") else None
        return (rc, mt)

    # Collect version dirs that look like "10.0.xxxxx.x"
    versions = [
        p for p in base.iterdir() if p.is_dir() and p.name and p.name[0].isdigit()
    ]
    if versions:
        # sort versions lexicographically by dotted parts (highest first)
        def ver_key(p: Path):
            parts = []
            for part in p.name.split("."):
                try:
                    parts.append(int(part))
                except ValueError:
                    parts.append(part)
            return parts

        versions.sort(key=ver_key, reverse=True)
        for v in versions:
            x64 = v / "x64"
            rc = x64 / "rc.exe"
            mt = x64 / "mt.exe"
            if rc.exists() and mt.exists():
                return (rc, mt)

    # Last-resort: search recursively and PATH lookup
    rc = next(base.rglob("rc.exe"), None) if base.exists() else None
    mt = next(base.rglob("mt.exe"), None) if base.exists() else None
    if not rc and shutil.which("rc.exe"):
        rc = Path(shutil.which("rc.exe"))
    if not mt and shutil.which("mt.exe"):
        mt = Path(shutil.which("mt.exe"))
    return (rc, mt)


def setup_environment(rocm_path: Path) -> Path:
    """Set required ROCm-related environment variables and print summary.

    Returns:
        Path: the computed CMake prefix path (rocm_path / "lib" / "cmake")
    """
    if not rocm_path.exists():
        sys.exit(f"Error: ROCM_PATH '{rocm_path}' does not exist.")

    cmake_prefix = directory_if_exists(rocm_path) / "lib" / "cmake"

    rocm_sdk_version = get_rocm_sdk_version() or "unknown"

    # TODO(#1658): when this gets merged into main consider refactoring environment setup
    env: dict[str, str] = {
        "ROCM_PATH": str(rocm_path),
        "ROCM_HOME": str(rocm_path),
        "HIP_PATH": str(rocm_path),
        "HIP_PATH_70": str(rocm_path),
        "LLVM_PATH": str(rocm_path),
        # "CMAKE_PREFIX_PATH": str(cmake_prefix),
    }

    # Workaround missing device lib bitcode. Same comment as in
    # pytorch/build_prod_wheels.py applies here. This is bad/annoying.
    # Unfortunately, it has been hardcoded for a long time. So we use a clang
    # env var to force a specific device lib path to workaround the hack to
    # get pytorch to build. This may or may not only affect the Python wheels
    # with their own quirks on directory layout. Obviously, this should be
    # completely burned with fire once the root causes are eliminted.
    hip_device_lib_path = rocm_path / "lib" / "llvm" / "amdgcn" / "bitcode"
    if not hip_device_lib_path.exists():
        print(
            "WARNING: Default location of device libs not found. Relying on "
            "clang heuristics which are known to be buggy in this configuration"
        )
    else:
        env["HIP_DEVICE_LIB_PATH"] = str(hip_device_lib_path)

    # find HIP clang
    hip_clang = directory_if_exists(rocm_path / "llvm" / "bin" / "clang")
    if not hip_clang:
        hip_clang = directory_if_exists(
            rocm_path / "lib" / "llvm" / "bin" / "clang.exe"
        )

    if not hip_clang:
        sys.exit(f"Error: HIP compiler not found at {hip_clang}")
    print(f"Using HIPCXX found at: {hip_clang}")
    env["HIPCXX"] = str(hip_clang)

    if os.name != "posix":
        llvm_dir = rocm_path / "lib" / "llvm" / "bin"
        env.update(
            {
                "HIP_CLANG_PATH": str(llvm_dir.resolve().as_posix()),
                "CC": str((llvm_dir / "clang-cl.exe").resolve()),
                "CXX": str((llvm_dir / "clang-cl.exe").resolve()),
            }
        )
    else:
        env.update(
            {
                # Workaround GCC12 compiler flags.
                "CXXFLAGS": " -Wno-error=maybe-uninitialized -Wno-error=uninitialized -Wno-error=restrict ",
                "CPPFLAGS": " -Wno-error=maybe-uninitialized -Wno-error=uninitialized -Wno-error=restrict ",
            }
        )

    # update PATH and environment
    # use correct separator for platform
    sep = ";" if os.name != "posix" else ":"
    os.environ[
        "PATH"
    ] = f"{str(rocm_path / 'bin')}{sep}{str(rocm_path / 'lib')}{sep}" + os.environ.get(
        "PATH", ""
    )
    os.environ.update(env)

    # Print summary information here (moved out of build)
    print(f"\nrocm version {rocm_sdk_version}:")
    print(f"  PYTHON VERSION: {sys.version}")
    print(f"  CMAKE_PREFIX_PATH = {cmake_prefix}")
    print(f"  ROCM_HOME = {rocm_path}")
    print("Environment variables set for ROCm:")
    for k, v in env.items():
        print(f"  {k}={v}")

    return cmake_prefix


def build(args: argparse.Namespace):
    rocm_dir: Path | None = args.rocm_dir
    repo_path: Path | None = args.llama_dir
    build_dir: Path | None = args.build_dir

    if not repo_path.exists():
        sys.exit(
            f"Error: Repository path '{repo_path}' does not exist. Run the llamacpp_repo.py script first."
        )

    # Setup environment and print summary from one place
    cmake_prefix = setup_environment(rocm_dir)

    # clean if requested
    if args.clean and build_dir.exists():
        print(f"Cleaning build directory ...")
        remove_dir_if_exists(build_dir)

    build_dir.mkdir(parents=True, exist_ok=True)

    rocm_targets = args.rocm_targets
    if rocm_targets is None:
        rocm_targets = get_rocm_sdk_targets()
        print(f"  Using default target from rocm-sdk targets: {rocm_targets}")
    else:
        print(f"  Using provided rocm targets: {rocm_targets}")

    if not rocm_targets:
        raise ValueError(
            "No --rocm-targets provided and rocm-sdk targets returned empty. "
            "Please specify --rocm-targets (e.g., gfx1100)."
        )

    rocm_targets_list = [t.strip() for t in rocm_targets.split(",")]
    rocm_targets_str = ";".join(rocm_targets_list)  # CMake expects ;-separated

    cmake_args = [
        "cmake",
        "-S",
        str(repo_path),
        "-B",
        str(build_dir),
        "-G",
        "Ninja",
        f"-DGPU_TARGETS={rocm_targets_str}",
        # Used to specify AMD GPU targets for older llama.cpp
        # repos that use GGML_AMDGPU_TARGETS instead of GPU_TARGETS
        f"-DAMDGPU_TARGETS={rocm_targets_str}",
        f"-DCMAKE_BUILD_TYPE={args.cmake_build_type}",
        f"-DLLAMA_CURL={args.llama_curl}",
        f"-DLLAMA_OPENSSL={args.llama_openssl}",
        f"-DLLAMA_LLGUIDANCE={args.llama_llguidance}",
        "-DHIP_PLATFORM=amd",
        "-DGGML_HIP=ON",
        # Windows-only RC/MT flags will be appended below when applicable
        # Disable rocWMMA fused attention to avoid build issues
        # TODO: re-evaluate this when rocWMMA is introduced in TheRock
        "-DGGML_HIP_ROCWMMA_FATTN=OFF",
        # Fixes the RPATH to find ROCm shared libraries at runtime
        # (libhipblas.so.3) TODO: check if needed for Windows
        f"-DCMAKE_BUILD_RPATH={str(rocm_dir / 'lib')}",
        f"-DCMAKE_INSTALL_RPATH={str(rocm_dir / 'lib')}",
    ]

    # Append Windows-specific compiler tools if running on Windows and found
    if os.name != "posix":
        rc_path, mt_path = find_windows_kits_tools()
        if rc_path and mt_path:
            print(f"Found Windows Kits tools: rc={rc_path}, mt={mt_path}")
            cmake_args.append(f"-DCMAKE_RC_COMPILER={rc_path.as_posix()}")
            cmake_args.append(f"-DCMAKE_MT={mt_path.as_posix()}")
        else:
            print(
                "Warning: rc.exe / mt.exe not found in Windows Kits. Skipping -DCMAKE_RC_COMPILER / -DCMAKE_MT flags."
            )

    if args.use_ccache:
        print("Using ccache, compilation results will be cached.")
        cmake_args.append("-DGGML_CCACHE=ON")
    else:
        cmake_args.append("-DGGML_CCACHE=OFF")

    print("\nCMake flags set:")
    for arg in cmake_args:
        if arg.startswith("-D"):
            print(f"  {arg}")

    print(f"\n++ Running CMake configuration for llama.cpp in {build_dir} ...")
    subprocess.run(cmake_args, check=True)

    print(f"\n++ Building llama.cpp in {build_dir} ...")
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


def main(argv: list[str]):
    p = argparse.ArgumentParser(description="Build llama.cpp with ROCm support")

    p.add_argument(
        "--rocm-dir",
        type=Path,
        default=get_rocm_path("root"),
        help="Path to ROCm installation directory",
    )
    p.add_argument(
        "--llama-dir",
        type=Path,
        default=directory_if_exists(script_dir / "llama.cpp"),
        help="Path to llama.cpp source directory",
    )
    p.add_argument(
        "--build-dir",
        type=Path,
        default=directory_if_exists(script_dir / "llama.cpp") / "build",
        help="Path to build directory",
    )
    p.add_argument(
        "--cmake-build-type",
        type=str,
        default="Release",
        help="CMake build type (Debug, Release, RelWithDebInfo, MinSizeRel)",
    )
    p.add_argument(
        "--rocm-targets",
        type=str,
        help="Comma-separated ROCm targets",
    )
    p.add_argument(
        "--jobs",
        type=str,
        default="0",
        help="Number of jobs for building (passed to cmake --build -j)",
    )
    p.add_argument(
        "--llama-curl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable libcurl support",
    )
    p.add_argument(
        "--llama-openssl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable OpenSSL support",
    )
    p.add_argument(
        "--llama-llguidance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable LLGuidance support",
    )
    p.add_argument(
        "--use-ccache",
        action=argparse.BooleanOptionalAction,
        help="Use ccache as the compiler launcher",
    )
    p.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Clean build directory before building",
    )

    args = p.parse_args(argv)

    print("\nArguments received:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("\nStarting build process...\n")

    build(args)


if __name__ == "__main__":
    main(sys.argv[1:])
