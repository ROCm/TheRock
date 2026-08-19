#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TheRock Multi-Environment & Modular Build Orchestrator.

Manages:
1. Python virtual environments per version (3.14, 3.13, 3.12, etc.) using `uv`.
2. Multiple independent modular ROCm builds within/across any Python virtual environment.
3. Hermetic installation of ROCm binaries directly into virtual environments (overriding system /opt/rocm).
4. Auto-patching of `venv/bin/activate` with ROCm paths for seamless activation.
5. Convenient component customization flags (--with-miopen, --with-profiler, etc.).
6. Batch matrix builds (building multiple presets sequentially).
7. Inspecting and activating specific build/environment pairs.
"""

import argparse
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
VENV_BASE_DIR = REPO_ROOT.parent.parent if REPO_ROOT.parent.name.startswith("venv") else Path.home() / "virtualenv"

# Predefined modular presets for experiments
PRESETS = {
    "llm": {
        "name": "llm",
        "description": "LLM Inference & LoRA/QLoRA Fine-Tuning (vLLM, llama.cpp, Ollama, Unsloth, HuggingFace) (~30-35m)",
        "cmake_flags": [
            "-DTHEROCK_ENABLE_ALL=OFF",
            "-DTHEROCK_ENABLE_COMPILER=ON",
            "-DTHEROCK_ENABLE_CORE=ON",
            "-DTHEROCK_ENABLE_CORE_RUNTIME=ON",
            "-DTHEROCK_ENABLE_HIP_RUNTIME=ON",
            "-DTHEROCK_ENABLE_CORE_AMDSMI=ON",
            "-DTHEROCK_ENABLE_BLAS=ON",
            "-DTHEROCK_ENABLE_PRIM=ON",
            "-DTHEROCK_ENABLE_RAND=ON",
            "-DTHEROCK_ENABLE_HIPTENSOR=ON",
        ],
    },
    "ai-full": {
        "name": "ai-full",
        "description": "Full AI Training & Vision/CNN Stack (PyTorch Training, MIOpen, Composable Kernel, RCCL, hipDNN) (~3.5-4.5h)",
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
    "vulkan-media": {
        "name": "vulkan-media",
        "description": "Graphics & Video Codecs (AMD Mesa Vulkan, rocDecode 4K/8K, rocJPEG, RPP) (~25-35m)",
        "cmake_flags": [
            "-DTHEROCK_ENABLE_ALL=OFF",
            "-DTHEROCK_ENABLE_COMPILER=ON",
            "-DTHEROCK_ENABLE_CORE=ON",
            "-DTHEROCK_ENABLE_HIP_RUNTIME=ON",
            "-DTHEROCK_ENABLE_SYSDEPS_AMD_MESA=ON",
            "-DTHEROCK_ENABLE_MEDIA_LIBS=ON",
            "-DTHEROCK_ENABLE_ROCDECODE=ON",
            "-DTHEROCK_ENABLE_ROCJPEG=ON",
            "-DTHEROCK_ENABLE_RPP=ON",
        ],
    },
    "core-hip": {
        "name": "core-hip",
        "description": "Minimal Foundation Engine (AMD Clang 23 Compiler + HIP Runtime + rocminfo) (~20-25m)",
        "cmake_flags": [
            "-DTHEROCK_ENABLE_ALL=OFF",
            "-DTHEROCK_ENABLE_COMPILER=ON",
            "-DTHEROCK_ENABLE_CORE=ON",
            "-DTHEROCK_ENABLE_CORE_RUNTIME=ON",
            "-DTHEROCK_ENABLE_HIP_RUNTIME=ON",
            "-DTHEROCK_ENABLE_CORE_AMDSMI=ON",
        ],
    },
    "math-hpc": {
        "name": "math-hpc",
        "description": "Scientific & HPC Math (rocBLAS, hipBLASLt, rocFFT, rocSOLVER, rocSPARSE, rocALUTION, rocRAND)",
        "cmake_flags": [
            "-DTHEROCK_ENABLE_ALL=OFF",
            "-DTHEROCK_ENABLE_COMPILER=ON",
            "-DTHEROCK_ENABLE_CORE=ON",
            "-DTHEROCK_ENABLE_HIP_RUNTIME=ON",
            "-DTHEROCK_ENABLE_MATH_LIBS=ON",
            "-DTHEROCK_ENABLE_ROCALUTION=ON",
        ],
    },
    "profiler": {
        "name": "profiler",
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
    "hipify": {
        "name": "hipify",
        "description": "CUDA to HIP Translation Tools (HIPIFY + Clang)",
        "cmake_flags": [
            "-DTHEROCK_ENABLE_ALL=OFF",
            "-DTHEROCK_ENABLE_COMPILER=ON",
            "-DTHEROCK_ENABLE_CORE=ON",
            "-DTHEROCK_ENABLE_HIP_RUNTIME=ON",
            "-DTHEROCK_ENABLE_HIPIFY=ON",
        ],
    },
    "opencl": {
        "name": "opencl",
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
        "name": "full",
        "description": "Complete Monolithic ROCm Stack (All 50+ libraries and tools) (~4.5-5.5h)",
        "cmake_flags": [
            "-DTHEROCK_ENABLE_ALL=ON",
        ],
    },
}

# Aliases for convenience
PRESET_ALIASES = {
    "llm-inference": "llm",
    "lora": "llm",
    "finetuning": "llm",
    "inference": "llm",
    "ai": "ai-full",
    "training": "ai-full",
    "pytorch": "ai-full",
    "vulkan": "vulkan-media",
    "media": "vulkan-media",
    "vision": "vulkan-media",
    "cv": "vulkan-media",
    "cv-vision": "vulkan-media",
    "hip": "core-hip",
    "core": "core-hip",
    "minimal": "core-hip",
    "math": "math-hpc",
    "hpc": "math-hpc",
    "scientific": "math-hpc",
}


def log_info(msg: str):
    print(f"\033[1;34m[INFO]\033[0m {msg}")


def log_success(msg: str):
    print(f"\033[1;32m[SUCCESS]\033[0m {msg}")


def log_warning(msg: str):
    print(f"\033[1;33m[WARNING]\033[0m {msg}")


def log_error(msg: str):
    print(f"\033[1;31m[ERROR]\033[0m {msg}")


def find_uv() -> str:
    """Find uv executable."""
    for loc in [shutil.which("uv"), str(Path.home() / ".local/bin/uv"), str(Path.home() / ".cargo/bin/uv")]:
        if loc and Path(loc).is_file():
            return loc
    log_error("uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh")
    sys.exit(1)


def get_venv_path(python_version: str) -> tuple[Path, Path]:
    """Get the parent directory and actual .venv directory for a python version."""
    py_slug = python_version.replace(".", "")
    parent_dir = VENV_BASE_DIR / f"venv{py_slug}"
    venv_dir = parent_dir / f".venv{py_slug}"
    return parent_dir, venv_dir


def detect_gpu_arch() -> str:
    """Detect local GPU architecture."""
    # Check existing build rocminfo
    for p in REPO_ROOT.glob("build*/dist/rocm/bin/rocminfo"):
        try:
            res = subprocess.run([str(p)], capture_output=True, text=True, timeout=5)
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    if "gfx" in line and "Name:" in line:
                        arch = line.split("Name:")[-1].strip()
                        if arch.startswith("gfx"):
                            return arch
        except Exception:
            pass

    # Check lspci
    try:
        res = subprocess.run(["lspci"], capture_output=True, text=True)
        if "8060S" in res.stdout or "8050S" in res.stdout or "Strix" in res.stdout:
            return "gfx1151"
    except Exception:
        pass

    return "gfx1151"


def ensure_submodules():
    """Ensure Git submodules (rocm-systems, rocm-libraries, etc.) are initialized with fast shallow clone."""
    hip_ver = REPO_ROOT / "rocm-systems/projects/hip/VERSION"
    if not hip_ver.is_file():
        log_info("Git submodules not found. Checking for local submodule cache...")
        # Check if another TheRock clone exists locally to avoid downloading GBs over network
        for candidate in Path.home().glob("virtualenv/**/TheRock"):
            if candidate != REPO_ROOT and (candidate / "rocm-systems/projects/hip/VERSION").is_file():
                log_info(f"Reusing existing local submodules from: {candidate}")
                for sm in ["rocm-systems", "rocm-libraries", "third-party"]:
                    src_sm = candidate / sm
                    dst_sm = REPO_ROOT / sm
                    if src_sm.is_dir() and not (dst_sm / ".git").exists():
                        shutil.copytree(src_sm, dst_sm, dirs_exist_ok=True)
                if hip_ver.is_file():
                    log_success("Local submodules linked successfully.")
                    return

        log_info("Initializing top-level git submodules using fast shallow clone (--depth 1)...")
        subprocess.check_call(["git", "submodule", "update", "--init", "--depth", "1", "rocm-systems", "rocm-libraries", "third-party"], cwd=str(REPO_ROOT))
        log_success("Git submodules initialized successfully.")


def apply_runtime_patches():
    """Apply necessary GCC 15 and Ubuntu 26.04 compatibility patches in-place to submodules."""
    # 1. MIOpen <ciso646> -> <version> fix
    miopen_header = REPO_ROOT / "rocm-libraries/projects/miopen/src/include/miopen/serializable.hpp"
    if miopen_header.is_file():
        content = miopen_header.read_text()
        if "#include <ciso646>" in content and "#if __has_include(<version>)" not in content:
            log_info("Applying GCC 15 <version> compatibility patch to MIOpen...")
            patched = content.replace(
                "#include <ciso646>",
                "#if __has_include(<version>)\n#include <version>\n#else\n#include <ciso646>\n#endif",
            )
            miopen_header.write_text(patched)

    # 2. rocprofiler-sdk sqlite3 public linkage fix
    rocprof_cmake = REPO_ROOT / "rocm-systems/projects/rocprofiler-sdk/source/lib/output/CMakeLists.txt"
    if rocprof_cmake.is_file():
        content = rocprof_cmake.read_text()
        if "PRIVATE\n            rocprofiler-sdk::rocprofiler-sdk-sqlite3" in content:
            log_info("Applying SQLite3 PUBLIC linkage patch to rocprofiler-sdk...")
            patched = content.replace(
                "PRIVATE\n            rocprofiler-sdk::rocprofiler-sdk-sqlite3",
                "PUBLIC\n            rocprofiler-sdk::rocprofiler-sdk-sqlite3",
            )
            rocprof_cmake.write_text(patched)


def ensure_venv(python_version: str, venv_dir: Path | None = None, force_recreate: bool = False) -> tuple[Path, Path]:
    """Ensure a virtual environment for the given Python version exists with dependencies installed."""
    uv_bin = find_uv()
    if venv_dir is None:
        parent_dir, venv_dir = get_venv_path(python_version)
    else:
        parent_dir = venv_dir.parent
    venv_python = venv_dir / "bin/python3" if not platform.system() == "Windows" else venv_dir / "Scripts/python.exe"

    if venv_python.exists() and not force_recreate:
        log_info(f"Reusing existing Python {python_version} virtualenv: \033[1;36m{venv_dir}\033[0m")
        return venv_dir, venv_python

    log_info(f"Creating Python {python_version} virtual environment using uv at: {venv_dir}")
    parent_dir.mkdir(parents=True, exist_ok=True)

    cmd_create = [uv_bin, "venv", str(venv_dir), "--python", python_version, "--allow-existing"]
    if force_recreate:
        cmd_create.append("--clear")
    subprocess.check_call(cmd_create)

    # Install requirements
    req_path = REPO_ROOT / "requirements.txt"
    if req_path.is_file():
        log_info(f"Installing build requirements using uv into Python {python_version} venv...")
        cmd_install = [uv_bin, "pip", "install", "--python", str(venv_python), "-r", str(req_path)]
        subprocess.check_call(cmd_install)

    log_success(f"Python {python_version} virtual environment ready at: {venv_dir}")
    return venv_dir, venv_python


def install_rocm_to_venv(build_dir: Path, venv_dir: Path, preset_name: str = "custom", py_ver: str = "3.14"):
    """Hermetically install ROCm binaries and environment hooks into the virtual environment."""
    rocm_dist = build_dir / "dist/rocm"
    if not rocm_dist.is_dir():
        log_warning(f"ROCm dist directory not found at {rocm_dist}. Skipping venv installation.")
        return

    log_info(f"Installing ROCm binaries & isolation wrappers into virtualenv: \033[1;36m{venv_dir}\033[0m")
    venv_bin = venv_dir / "bin"
    venv_bin.mkdir(parents=True, exist_ok=True)

    rocm_bin_dirs = [
        rocm_dist / "bin",
        rocm_dist / "lib/llvm/bin",
    ]

    installed_bins = 0
    # Create hermetic wrapper scripts for all binaries
    for b_dir in rocm_bin_dirs:
        if not b_dir.is_dir():
            continue
        for src_bin in b_dir.iterdir():
            # Skip directories or yaml files
            if src_bin.is_dir() or src_bin.name.endswith(".yaml") or src_bin.name.endswith(".py"):
                continue
            
            target_bin = venv_bin / src_bin.name
            
            # Create bash wrapper that sets environment and runs binary
            wrapper_content = f"""#!/bin/bash
# Hermetic ROCm wrapper auto-generated by TheRock [{preset_name}]
export ROCM_PATH="{rocm_dist}"
export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
export HIP_DEVICE_LIB_PATH="$ROCM_PATH/lib/llvm/amdgcn/bitcode"
export CMAKE_PREFIX_PATH="$ROCM_PATH:$ROCM_PATH/lib/cmake${{CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}}"

exec "{src_bin.resolve()}" "$@"
"""
            target_bin.write_text(wrapper_content)
            target_bin.chmod(0o755)
            installed_bins += 1

    log_success(f"Installed {installed_bins} isolated ROCm executable wrappers into {venv_bin}")

    # Patch venv/bin/activate to automatically export ROCm environment variables
    activate_file = venv_bin / "activate"
    if activate_file.is_file():
        act_text = activate_file.read_text()
        
        hook_marker_start = "# --- BEGIN THEROCK ROCM ENVIRONMENT ---"
        hook_marker_end = "# --- END THEROCK ROCM ENVIRONMENT ---"
        
        hook_block = f"""{hook_marker_start}
# Auto-injected by TheRock for [{preset_name} / Python {py_ver}]
_OLD_THEROCK_ROCM_PATH="$ROCM_PATH"
_OLD_THEROCK_LD_PATH="$LD_LIBRARY_PATH"
_OLD_THEROCK_HIP_LIB_PATH="$HIP_DEVICE_LIB_PATH"

export ROCM_PATH="{rocm_dist}"
export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
export HIP_DEVICE_LIB_PATH="$ROCM_PATH/lib/llvm/amdgcn/bitcode"
export CMAKE_PREFIX_PATH="$ROCM_PATH:$ROCM_PATH/lib/cmake${{CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}}"
{hook_marker_end}"""

        # Replace existing hook if present, else append
        if hook_marker_start in act_text:
            act_text = re.sub(
                f"{re.escape(hook_marker_start)}.*?{re.escape(hook_marker_end)}",
                hook_block,
                act_text,
                flags=re.DOTALL,
            )
        else:
            act_text += f"\n\n{hook_block}\n"

        # Also patch deactivate function to restore old ROCm variables if not already patched
        if "deactivate () {" in act_text and "_OLD_THEROCK_ROCM_PATH" not in act_text.split("deactivate () {")[1].split("}")[0]:
            deact_restore = """    # Reset TheRock ROCm variables
    if [ -n "${_OLD_THEROCK_ROCM_PATH+x}" ]; then
        ROCM_PATH="$_OLD_THEROCK_ROCM_PATH"
        export ROCM_PATH
        unset _OLD_THEROCK_ROCM_PATH
    else
        unset ROCM_PATH
    fi
    if [ -n "${_OLD_THEROCK_LD_PATH+x}" ]; then
        LD_LIBRARY_PATH="$_OLD_THEROCK_LD_PATH"
        export LD_LIBRARY_PATH
        unset _OLD_THEROCK_LD_PATH
    else
        unset LD_LIBRARY_PATH
    fi
    if [ -n "${_OLD_THEROCK_HIP_LIB_PATH+x}" ]; then
        HIP_DEVICE_LIB_PATH="$_OLD_THEROCK_HIP_LIB_PATH"
        export HIP_DEVICE_LIB_PATH
        unset _OLD_THEROCK_HIP_LIB_PATH
    else
        unset HIP_DEVICE_LIB_PATH
    fi
"""
            act_text = act_text.replace("deactivate () {", f"deactivate () {{\n{deact_restore}")

        activate_file.write_text(act_text)
        log_success(f"Patched {activate_file} to automatically activate this ROCm build!")


def generate_activation_script(build_dir: Path, venv_dir: Path, preset_name: str, py_ver: str):
    """Generate activate_env.sh in build directory."""
    build_dir.mkdir(parents=True, exist_ok=True)
    script_path = build_dir / "activate_env.sh"
    
    content = f"""#!/bin/bash
# Environment Activation Script for TheRock [{preset_name} / Python {py_ver}]
# Usage: source {script_path.name}

BUILD_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
export ROCM_PATH="$BUILD_DIR/dist/rocm"

# 1. Activate Python {py_ver} Virtual Environment
if [ -f "{venv_dir}/bin/activate" ]; then
    source "{venv_dir}/bin/activate"
fi

# 2. Export ROCm Toolchain and Library Paths
if [ -d "$ROCM_PATH" ]; then
    export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$PATH"
    export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
    export HIP_DEVICE_LIB_PATH="$ROCM_PATH/lib/llvm/amdgcn/bitcode"
    export CMAKE_PREFIX_PATH="$ROCM_PATH:$ROCM_PATH/lib/cmake${{CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}}"
fi

echo -e "\\033[1;32m[TheRock Activated]\\033[0m"
echo -e "  Preset      : \\033[1;36m{preset_name}\\033[0m"
echo -e "  Python      : \\033[1;36m$(which python3)\\033[0m"
echo -e "  ROCM_PATH   : \\033[1;36m$ROCM_PATH\\033[0m"
echo -e "  rocminfo    : \\033[1;36m$(which rocminfo 2>/dev/null || echo 'not in path')\\033[0m"
"""
    script_path.write_text(content)
    script_path.chmod(0o755)
    log_success(f"Generated activation script: {script_path}")


def cmd_setup_env(args):
    """Command: Create/update Python virtual environment."""
    for py_ver in args.python:
        ensure_venv(py_ver, force_recreate=args.recreate)


def cmd_list_envs(args):
    """Command: List all detected Python virtual environments."""
    print("\n\033[1;37mDetected Virtual Environments (in ~/virtualenv/):\033[0m")
    print("-" * 65)
    print(f"{'Directory':<20} | {'Python Executable':<30} | {'Status':<10}")
    print("-" * 65)
    
    if not VENV_BASE_DIR.is_dir():
        print("  (No virtualenv base directory found)")
        return

    found = 0
    for item in sorted(VENV_BASE_DIR.glob("venv*")):
        if item.is_dir():
            dot_venvs = list(item.glob(".venv*"))
            if dot_venvs:
                for dv in dot_venvs:
                    py_bin = dv / "bin/python3"
                    status = "\033[1;32mActive\033[0m" if py_bin.exists() else "\033[1;31mMissing\033[0m"
                    print(f"{dv.parent.name + '/' + dv.name:<20} | {str(py_bin):<30} | {status:<10}")
                    found += 1
            else:
                py_bin = item / "bin/python3"
                if py_bin.exists():
                    print(f"{item.name:<20} | {str(py_bin):<30} | \033[1;32mActive\033[0m")
                    found += 1
    if found == 0:
        print("  (No virtualenvs created yet. Create one with: ./therock-env setup-venv 3.14)")
    print("-" * 65)


def cmd_list_builds(args):
    """Command: List all completed build trees and their presets."""
    print("\n\033[1;37mTheRock Build Trees:\033[0m")
    print("-" * 75)
    print(f"{'Build Directory':<22} | {'ROCm Dist':<10} | {'Size':<10} | {'Activation Script'}")
    print("-" * 75)

    build_dirs = sorted(REPO_ROOT.glob("build*"))
    found = 0
    for b in build_dirs:
        if b.is_dir() and b.name != "build_tools":
            dist_dir = b / "dist/rocm"
            act_script = b / "activate_env.sh"
            has_dist = "\033[1;32mReady\033[0m" if (dist_dir / "bin/rocminfo").exists() else "\033[1;33mPartial\033[0m"
            
            # Size estimate
            try:
                du_res = subprocess.run(["du", "-sh", str(b)], capture_output=True, text=True, timeout=5)
                size_str = du_res.stdout.split()[0] if du_res.returncode == 0 else "-"
            except Exception:
                size_str = "-"
            
            act_str = str(act_script.relative_to(REPO_ROOT)) if act_script.is_file() else "-"
            print(f"{b.name:<22} | {has_dist:<10} | {size_str:<10} | {act_str}")
            found += 1

    if found == 0:
        print("  (No build directories found)")
    print("-" * 75)


def cmd_install_to_venv(args):
    """Command: Hermetically install an existing ROCm build into a virtualenv."""
    build_dir = REPO_ROOT / args.build_dir
    parent_dir, venv_dir = get_venv_path(args.python)
    install_rocm_to_venv(build_dir, venv_dir, preset_name="installed", py_ver=args.python)


def cmd_build(args):
    """Command: Configure and build a specific preset in a Python environment."""
    preset_key = PRESET_ALIASES.get(args.preset, args.preset)
    if preset_key not in PRESETS:
        log_error(f"Unknown preset: {args.preset}. Available: {', '.join(PRESETS.keys())}")
        sys.exit(1)

    preset_data = PRESETS[preset_key]
    py_ver = args.python
    gpu_arch = args.gpu_target or detect_gpu_arch()

    # Determine build directory
    py_slug = py_ver.replace(".", "")
    preset_slug = preset_key.replace("-", "_")
    build_dir_name = args.build_dir or f"build_py{py_slug}_{preset_slug}"
    build_dir = REPO_ROOT / build_dir_name

    log_info(f"============================================================")
    log_info(f"Target Preset     : \033[1;36m{preset_key}\033[0m ({preset_data['description']})")
    log_info(f"Python Version    : \033[1;36m{py_ver}\033[0m")
    log_info(f"GPU Architecture  : \033[1;32m{gpu_arch}\033[0m")
    log_info(f"Build Directory   : \033[1;36m{build_dir}\033[0m")
    log_info(f"============================================================")

    # 0. Ensure git submodules are present and patches applied
    ensure_submodules()
    apply_runtime_patches()

    # 1. Setup / ensure virtualenv
    custom_venv_dir = getattr(args, "venv_dir", None)
    venv_dir, venv_python = ensure_venv(py_ver, venv_dir=custom_venv_dir)

    # 2. Setup CMake Flags
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

    # Process custom --with-* and --without-* flags
    if getattr(args, "with_miopen", False):
        log_info("Custom Flag: Enabling MIOpen (+Composable Kernel)")
        cmake_cmd.append("-DTHEROCK_ENABLE_MIOPEN=ON")
    if getattr(args, "with_rccl", False):
        log_info("Custom Flag: Enabling RCCL (Collective Communications)")
        cmake_cmd.append("-DTHEROCK_ENABLE_RCCL=ON")
    if getattr(args, "with_profiler", False):
        log_info("Custom Flag: Enabling Profiler & Debug Tools")
        cmake_cmd.extend(["-DTHEROCK_ENABLE_PROFILER=ON", "-DTHEROCK_ENABLE_ROCPROFV3=ON", "-DTHEROCK_ENABLE_ROCGDB=ON"])
    if getattr(args, "with_fft", False):
        log_info("Custom Flag: Enabling rocFFT")
        cmake_cmd.append("-DTHEROCK_ENABLE_MATH_LIBS=ON")
    if getattr(args, "with_media", False) or getattr(args, "with_vulkan", False):
        log_info("Custom Flag: Enabling Mesa / Vulkan / Video Codecs")
        cmake_cmd.extend([
            "-DTHEROCK_ENABLE_SYSDEPS_AMD_MESA=ON",
            "-DTHEROCK_ENABLE_MEDIA_LIBS=ON",
            "-DTHEROCK_ENABLE_ROCDECODE=ON",
            "-DTHEROCK_ENABLE_ROCJPEG=ON",
        ])
    if getattr(args, "without_blas", False):
        log_info("Custom Flag: Disabling BLAS libraries")
        cmake_cmd.append("-DTHEROCK_ENABLE_BLAS=OFF")

    if not args.no_ccache:
        cmake_cmd.extend([
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        ])

    if args.extra_cmake_args:
        cmake_cmd.extend(args.extra_cmake_args)

    # 3. Configure
    env = os.environ.copy()
    env["PATH"] = f"{venv_dir}/bin:{env.get('PATH', '')}"

    log_info("Running CMake configuration...")
    if args.dry_run:
        log_info(f"[Dry-Run] {' '.join(cmake_cmd)}")
    else:
        start_t = time.time()
        subprocess.check_call(cmake_cmd, env=env, cwd=str(REPO_ROOT))
        log_success(f"CMake configuration finished in {time.time() - start_t:.1f}s")

    # 4. Generate activate script
    generate_activation_script(build_dir, venv_dir, preset_key, py_ver)

    # 5. Build
    if not args.configure_only and not args.dry_run:
        log_info(f"Starting Ninja build in {build_dir}...")
        start_build_t = time.time()
        subprocess.check_call(["ninja", "-C", str(build_dir)], env=env)
        elapsed_m = (time.time() - start_build_t) / 60
        log_success(f"Build '{preset_key}' finished successfully in {elapsed_m:.1f} minutes!")

        # 6. Hermetically install ROCm wrappers and patch venv/bin/activate
        install_rocm_to_venv(build_dir, venv_dir, preset_name=preset_key, py_ver=py_ver)

        print("\n\033[1;32m=========================================================\033[0m")
        print(f"\033[1;32mROCm '{preset_key}' is now hermetically installed into Python {py_ver} venv!\033[0m")
        print(f"To activate, simply run:")
        print(f"  \033[1;36msource {venv_dir}/bin/activate\033[0m")
        print(f"  or: \033[1;36msource {build_dir.name}/activate_env.sh\033[0m")
        print("\033[1;32m=========================================================\033[0m\n")


def cmd_build_matrix(args):
    """Command: Build multiple presets sequentially."""
    raw_presets = [p.strip() for p in args.presets.split(",")]
    log_info(f"Running matrix build for presets: {raw_presets} on Python {args.python}")

    for idx, raw_preset in enumerate(raw_presets, 1):
        preset_name = PRESET_ALIASES.get(raw_preset, raw_preset)
        log_info(f"\n>>> [{idx}/{len(raw_presets)}] Starting build for preset: {preset_name} <<<")
        args.preset = preset_name
        args.build_dir = None
        cmd_build(args)


def main():
    parser = argparse.ArgumentParser(
        description="TheRock Multi-Environment & Modular Build Orchestrator",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Command: setup-venv
    p_venv = subparsers.add_parser("setup-venv", help="Create or update Python virtual environments with uv")
    p_venv.add_argument("python", nargs="+", default=["3.14"], help="Python version(s) to create (e.g. 3.14 3.13)")
    p_venv.add_argument("--recreate", action="store_true", help="Force recreate virtual environment")
    p_venv.set_defaults(func=cmd_setup_env)

    # Command: list-envs
    p_list_envs = subparsers.add_parser("list-envs", help="List all detected Python virtual environments")
    p_list_envs.set_defaults(func=cmd_list_envs)

    # Command: list-builds
    p_list_builds = subparsers.add_parser("list-builds", help="List all completed and partial ROCm build trees")
    p_list_builds.set_defaults(func=cmd_list_builds)

    # Command: install-to-venv
    p_inst = subparsers.add_parser("install-to-venv", help="Install an existing ROCm build into a Python virtual environment")
    p_inst.add_argument("--build-dir", default="build_1151", help="Build directory containing dist/rocm (default: build_1151)")
    p_inst.add_argument("--python", default="3.14", help="Target Python virtual environment version (default: 3.14)")
    p_inst.set_defaults(func=cmd_install_to_venv)

    # Helper function to add build flags
    def add_build_options(p):
        p.add_argument("--python", default="3.14", help="Python version to target (default: 3.14)")
        p.add_argument("--gpu-target", default=None, help="GPU target (default: auto-detected, e.g. gfx1151)")
        p.add_argument("--venv-dir", type=Path, default=None, help="Custom virtual environment directory")
        p.add_argument("--build-dir", default=None, help="Custom build directory name")
        p.add_argument("--no-ccache", action="store_true", help="Disable ccache")
        p.add_argument("--configure-only", action="store_true", help="Only run CMake configure")
        p.add_argument("--dry-run", action="store_true", help="Print commands without executing")
        # Component toggle flags
        p.add_argument("--with-miopen", action="store_true", help="Add MIOpen (Deep Learning Convolutions) to build")
        p.add_argument("--with-rccl", action="store_true", help="Add RCCL (Multi-GPU Communications) to build")
        p.add_argument("--with-profiler", action="store_true", help="Add Profiler & rocgdb tools to build")
        p.add_argument("--with-fft", action="store_true", help="Add rocFFT to build")
        p.add_argument("--with-media", action="store_true", help="Add Mesa, rocDecode, rocJPEG to build")
        p.add_argument("--with-vulkan", action="store_true", help="Add AMD Mesa Vulkan to build")
        p.add_argument("--without-blas", action="store_true", help="Exclude BLAS math libraries")
        p.add_argument("--extra-cmake-args", nargs="*", default=[], help="Extra CMake options")

    # Command: build
    p_build = subparsers.add_parser("build", help="Build a modular ROCm preset in a Python virtualenv")
    p_build.add_argument("--preset", default="hip", help="Preset to build (e.g. hip, llm, vulkan, math, ai, hpc, cv)")
    add_build_options(p_build)
    p_build.set_defaults(func=cmd_build)

    # Command: build-matrix
    p_matrix = subparsers.add_parser("build-matrix", help="Build multiple presets sequentially")
    p_matrix.add_argument("--presets", default="hip,llm,vulkan", help="Comma-separated presets (e.g. hip,llm,vulkan)")
    add_build_options(p_matrix)
    p_matrix.set_defaults(func=cmd_build_matrix)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
