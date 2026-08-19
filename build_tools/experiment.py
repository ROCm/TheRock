#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TheRock Automated Modular Experiment & Virtual Environment Builder.

Automates:
1. Creating and managing isolated virtual environments with `uv` (for any Python version).
2. Configuring and building specific subsets/presets of ROCm (e.g. HIP-only, Vulkan/Media, Math, PyTorch AI).
3. Automatically detecting GPU architecture (e.g. gfx1151 for Strix Halo).
4. Enabling ccache acceleration and generating ready-to-use environment activation scripts.
"""

import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Predefined modular presets for experiments
PRESETS = {
    "hip-minimal": {
        "description": "Minimal HIP Runtime + Compiler + rocminfo (Fastest build ~20-30m)",
        "cmake_flags": [
            "-DTHEROCK_ENABLE_ALL=OFF",
            "-DTHEROCK_ENABLE_COMPILER=ON",
            "-DTHEROCK_ENABLE_CORE=ON",
            "-DTHEROCK_ENABLE_CORE_RUNTIME=ON",
            "-DTHEROCK_ENABLE_HIP_RUNTIME=ON",
            "-DTHEROCK_ENABLE_CORE_AMDSMI=ON",
        ],
    },
    "vulkan-media": {
        "description": "HIP + AMD Mesa (Vulkan/VAAPI) + rocDecode + rocJPEG (Video/Image acceleration)",
        "cmake_flags": [
            "-DTHEROCK_ENABLE_ALL=OFF",
            "-DTHEROCK_ENABLE_COMPILER=ON",
            "-DTHEROCK_ENABLE_CORE=ON",
            "-DTHEROCK_ENABLE_HIP_RUNTIME=ON",
            "-DTHEROCK_ENABLE_SYSDEPS_AMD_MESA=ON",
            "-DTHEROCK_ENABLE_MEDIA_LIBS=ON",
            "-DTHEROCK_ENABLE_ROCDECODE=ON",
            "-DTHEROCK_ENABLE_ROCJPEG=ON",
        ],
    },
    "math-blas": {
        "description": "HIP + Math Stack (rocBLAS, hipBLASLt, rocRAND, rocPRIM, rocThrust, rocSOLVER, rocFFT)",
        "cmake_flags": [
            "-DTHEROCK_ENABLE_ALL=OFF",
            "-DTHEROCK_ENABLE_COMPILER=ON",
            "-DTHEROCK_ENABLE_CORE=ON",
            "-DTHEROCK_ENABLE_HIP_RUNTIME=ON",
            "-DTHEROCK_ENABLE_MATH_LIBS=ON",
        ],
    },
    "ai-pytorch": {
        "description": "PyTorch / JAX AI Stack (HIP + Math + MIOpen + Composable Kernel + RCCL + hipDNN)",
        "cmake_flags": [
            "-DTHEROCK_ENABLE_ALL=OFF",
            "-DTHEROCK_ENABLE_COMPILER=ON",
            "-DTHEROCK_ENABLE_CORE=ON",
            "-DTHEROCK_ENABLE_HIP_RUNTIME=ON",
            "-DTHEROCK_ENABLE_MATH_LIBS=ON",
            "-DTHEROCK_ENABLE_ML_LIBS=ON",
            "-DTHEROCK_ENABLE_MIOPEN=ON",
            "-DTHEROCK_ENABLE_RCCL=ON",
        ],
    },
    "profiler": {
        "description": "ROCm Profiling & Debug Tools (rocprofiler-sdk, rocprofiler-systems, rocgdb, roctracer)",
        "cmake_flags": [
            "-DTHEROCK_ENABLE_ALL=OFF",
            "-DTHEROCK_ENABLE_COMPILER=ON",
            "-DTHEROCK_ENABLE_CORE=ON",
            "-DTHEROCK_ENABLE_HIP_RUNTIME=ON",
            "-DTHEROCK_ENABLE_DEBUG_TOOLS=ON",
            "-DTHEROCK_ENABLE_PROFILER=ON",
            "-DTHEROCK_ENABLE_ROCGDB=ON",
            "-DTHEROCK_ENABLE_ROCPROFV3=ON",
            "-DTHEROCK_ENABLE_ROCPROFSYS=ON",
        ],
    },
    "opencl": {
        "description": "OpenCL & HIP Runtimes (ocl-clr, hip-clr, OpenCL ICD)",
        "cmake_flags": [
            "-DTHEROCK_ENABLE_ALL=OFF",
            "-DTHEROCK_ENABLE_COMPILER=ON",
            "-DTHEROCK_ENABLE_CORE=ON",
            "-DTHEROCK_ENABLE_HIP_RUNTIME=ON",
            "-DTHEROCK_ENABLE_OCL_RUNTIME=ON",
        ],
    },
    "full": {
        "description": "Full Complete ROCm Stack (All 50+ libraries and tools)",
        "cmake_flags": [
            "-DTHEROCK_ENABLE_ALL=ON",
        ],
    },
}


def log_info(msg: str):
    print(f"\033[1;34m[INFO]\033[0m {msg}")


def log_success(msg: str):
    print(f"\033[1;32m[SUCCESS]\033[0m {msg}")


def log_warning(msg: str):
    print(f"\033[1;33m[WARNING]\033[0m {msg}")


def log_error(msg: str):
    print(f"\033[1;31m[ERROR]\033[0m {msg}")


def detect_gpu_arch() -> str:
    """Detect the local GPU architecture using rocminfo, lspci, or default to gfx1151."""
    # Check rocminfo in current path or existing dist
    for rocminfo_path in ["rocminfo", str(REPO_ROOT / "build_1151/dist/rocm/bin/rocminfo")]:
        try:
            res = subprocess.run([rocminfo_path], capture_output=True, text=True, timeout=5)
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    if "gfx" in line and "Name:" in line:
                        arch = line.split("Name:")[-1].strip()
                        if arch.startswith("gfx"):
                            return arch
        except Exception:
            pass

    # Check lspci for Strix Halo
    try:
        res = subprocess.run(["lspci"], capture_output=True, text=True)
        if "Strix" in res.stdout or "8060S" in res.stdout or "8050S" in res.stdout:
            return "gfx1151"
    except Exception:
        pass

    return "gfx1151"


def find_uv() -> str:
    """Find uv binary in PATH or default local locations."""
    uv_bin = shutil.which("uv")
    if uv_bin:
        return uv_bin
    
    home_uv = Path.home() / ".local/bin/uv"
    if home_uv.is_file():
        return str(home_uv)
    
    cargo_uv = Path.home() / ".cargo/bin/uv"
    if cargo_uv.is_file():
        return str(cargo_uv)
    
    log_error("`uv` executable not found. Please install via: curl -LsSf https://astral.sh/uv/install.sh | sh")
    sys.exit(1)


def setup_virtual_environment(uv_path: str, venv_dir: Path, python_ver: str, recreate: bool = False) -> Path:
    """Create or reuse virtual environment with uv and install requirements."""
    venv_python = venv_dir / "bin/python3" if not platform.system() == "Windows" else venv_dir / "Scripts/python.exe"
    
    if venv_python.exists() and not recreate:
        log_info(f"Reusing existing virtual environment: {venv_dir}")
        return venv_python

    log_info(f"Setting up Python {python_ver} virtual environment at: {venv_dir}")
    venv_dir.parent.mkdir(parents=True, exist_ok=True)

    # 1. Create venv with --allow-existing
    cmd_venv = [uv_path, "venv", str(venv_dir), "--python", python_ver, "--allow-existing"]
    log_info(f"Running: {' '.join(cmd_venv)}")
    subprocess.check_call(cmd_venv)

    if not venv_python.exists():
        log_error(f"Python binary not found in venv: {venv_python}")
        sys.exit(1)

    # 2. Install requirements using uv pip
    req_file = REPO_ROOT / "requirements.txt"
    if req_file.is_file():
        log_info("Installing build dependencies from requirements.txt...")
        cmd_pip = [uv_path, "pip", "install", "--python", str(venv_python), "-r", str(req_file)]
        subprocess.check_call(cmd_pip)

    log_success(f"Virtual environment ready: {venv_dir}")
    return venv_python


def generate_activation_script(build_dir: Path, venv_dir: Path):
    """Generate shell script to activate the exact environment and ROCm distribution."""
    build_dir.mkdir(parents=True, exist_ok=True)
    script_path = build_dir / "activate_env.sh"
    rocm_dist = build_dir / "dist/rocm"
    
    content = f"""#!/bin/bash
# Auto-generated by build_tools/experiment.py
# Source this file to activate both the Python virtualenv and the compiled ROCm stack:
#   source {script_path.relative_to(REPO_ROOT) if script_path.is_relative_to(REPO_ROOT) else script_path}

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
export ROCM_PATH="$SCRIPT_DIR/dist/rocm"

# Activate Python Virtual Environment
if [ -f "{venv_dir}/bin/activate" ]; then
    source "{venv_dir}/bin/activate"
fi

# Export ROCm Environment Paths
if [ -d "$ROCM_PATH" ]; then
    export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$PATH"
    export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
    export HIP_DEVICE_LIB_PATH="$ROCM_PATH/lib/llvm/amdgcn/bitcode"
    export CMAKE_PREFIX_PATH="$ROCM_PATH:$ROCM_PATH/lib/cmake${{CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}}"
    echo "[ROCm Environment Activated]"
    echo "  ROCM_PATH : $ROCM_PATH"
    echo "  Python    : $(which python3)"
fi
"""
    script_path.write_text(content)
    script_path.chmod(0o755)
    log_success(f"Generated environment activation script: {script_path}")


def main():
    parser = argparse.ArgumentParser(
        description="TheRock Modular Experiment & Virtual Environment Automation",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="hip-minimal",
        help="Experiment preset to build. Choices:\n"
        + "\n".join([f"  {k:15} : {v['description']}" for k, v in PRESETS.items()]),
    )
    parser.add_argument(
        "--python",
        default="3.14",
        help="Python version to use for virtual environment (default: 3.14)",
    )
    parser.add_argument(
        "--venv-dir",
        type=Path,
        default=None,
        help="Path for virtual environment (default: ~/virtualenv/venv<py>_<preset>)",
    )
    parser.add_argument(
        "--recreate-venv",
        action="store_true",
        help="Force re-creation of virtual environment even if it exists",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="Path for build directory (default: build_<preset>)",
    )
    parser.add_argument(
        "--gpu-target",
        default=None,
        help="AMDGPU target architecture (default: auto-detected, e.g. gfx1151)",
    )
    parser.add_argument(
        "--no-ccache",
        action="store_true",
        help="Disable ccache compiler launcher",
    )
    parser.add_argument(
        "--configure-only",
        action="store_true",
        help="Configure CMake project without running compilation",
    )
    parser.add_argument(
        "--fetch-sources",
        action="store_true",
        help="Force re-running fetch_sources.py to update git submodules",
    )
    parser.add_argument(
        "--extra-cmake-args",
        nargs="*",
        default=[],
        help="Extra arguments to pass directly to CMake",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configure and build commands without executing them",
    )

    args = parser.parse_args()

    # 1. Preset info
    preset_data = PRESETS[args.preset]
    log_info(f"Selected Preset: \033[1;36m{args.preset}\033[0m ({preset_data['description']})")

    # 2. Target GPU
    gpu_arch = args.gpu_target or detect_gpu_arch()
    log_info(f"Target GPU Architecture: \033[1;32m{gpu_arch}\033[0m")

    # 3. Virtual Environment Path
    py_slug = args.python.replace(".", "")
    preset_slug = args.preset.replace("-", "_")
    venv_dir = (
        args.venv_dir
        if args.venv_dir
        else Path.home() / f"virtualenv/venv{py_slug}_{preset_slug}"
    )

    # 4. Build Directory Path
    build_dir = args.build_dir if args.build_dir else REPO_ROOT / f"build_{preset_slug}"
    log_info(f"Build Directory: {build_dir}")

    # 5. Virtualenv creation with uv
    uv_path = find_uv()
    venv_python = setup_virtual_environment(uv_path, venv_dir, args.python, recreate=args.recreate_venv)

    # 6. Fetch sources using the venv python if needed
    if args.fetch_sources:
        log_info("Fetching submodules and applying patches...")
        subprocess.check_call([str(venv_python), str(REPO_ROOT / "build_tools/fetch_sources.py")])

    # 7. Configure CMake Flags
    cmake_cmd = [
        "cmake",
        "-B",
        str(build_dir),
        "-GNinja",
        "-S",
        str(REPO_ROOT),
        f"-DTHEROCK_AMDGPU_FAMILIES={gpu_arch}",
    ]
    cmake_cmd.extend(preset_data["cmake_flags"])

    # CCache setup
    if not args.no_ccache:
        cmake_cmd.extend([
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        ])

    if args.extra_cmake_args:
        cmake_cmd.extend(args.extra_cmake_args)

    # 8. Run CMake Configure
    log_info(f"Configuring with CMake:\n  {' '.join(cmake_cmd)}")
    if args.dry_run:
        log_info("[Dry-Run] Skipped CMake execution.")
    else:
        # Evaluate ccache env if setup_ccache exists
        env = os.environ.copy()
        env["PATH"] = f"{venv_dir}/bin:{env.get('PATH', '')}"
        
        start_time = time.time()
        subprocess.check_call(cmake_cmd, env=env, cwd=str(REPO_ROOT))
        log_success(f"CMake configuration completed in {time.time() - start_time:.1f}s")

    # 9. Generate environment activator
    generate_activation_script(build_dir, venv_dir)

    # 10. Run Build if requested
    if not args.configure_only and not args.dry_run:
        log_info(f"Starting build with Ninja in {build_dir}...")
        start_build_time = time.time()
        subprocess.check_call(["ninja", "-C", str(build_dir)], env=env)
        elapsed = time.time() - start_build_time
        log_success(f"Build succeeded in {elapsed / 60:.1f} minutes!")
        print()
        print("\033[1;32m===================================================\033[0m")
        print(f"\033[1;32mExperiment '{args.preset}' is ready to use!\033[0m")
        print(f"Activate with:  \033[1;36msource {build_dir}/activate_env.sh\033[0m")
        print("\033[1;32m===================================================\033[0m")


if __name__ == "__main__":
    main()
