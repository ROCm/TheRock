#!/usr/bin/env python3
"""
Standalone RBT Builder - Build ROCm Bandwidth Test

Usage:
  # Basic build (uses system ROCm at /opt/rocm)
  python build_tools/build_rbt_standalone.py

  # Build with custom ROCm path
  ROCM_PATH=/path/to/rocm python build_tools/build_rbt_standalone.py

  # Build with TheRock artifacts
  python build_tools/build_rbt_standalone.py --rocm-path ./build

  # Clean build
  python build_tools/build_rbt_standalone.py --clean

  # Debug build
  python build_tools/build_rbt_standalone.py --debug

Options:
  --rocm-path PATH    Path to ROCm installation (default: $ROCM_PATH or /opt/rocm)
  --install-path PATH Install location (default: same as rocm-path)
  --clean             Clean build directory before building
  --debug             Build debug version instead of release
  --no-install        Skip installation step
  --verbose           Show verbose CMake output
  --help              Show this help message
"""

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


def run_cmd(cmd, cwd=None, check=True, capture=False, env=None):
    """Execute a command"""
    log.info(f"$ {cmd}")
    
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    
    if capture:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd, env=merged_env,
            capture_output=True, text=True
        )
    else:
        result = subprocess.run(cmd, shell=True, cwd=cwd, env=merged_env)
    
    if check and result.returncode != 0:
        log.error(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    
    return result


def detect_local_gpus(rocm_dir):
    """Detect local GPU architectures using rocm_agent_enumerator"""
    enumerator = Path(rocm_dir) / "bin" / "rocm_agent_enumerator"
    
    if not enumerator.exists():
        log.warning(f"rocm_agent_enumerator not found at {enumerator}")
        return []
    
    try:
        result = subprocess.run(
            [str(enumerator)],
            capture_output=True,
            text=True,
            check=True
        )
        gpus = [line.strip() for line in result.stdout.split('\n') 
                if line.strip().startswith('gfx')]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_gpus = []
        for gpu in gpus:
            if gpu not in seen:
                seen.add(gpu)
                unique_gpus.append(gpu)
        
        if unique_gpus:
            log.info(f"Detected GPUs: {', '.join(unique_gpus)}")
            return unique_gpus
        else:
            log.warning("No GPUs detected")
            return []
    except Exception as e:
        log.warning(f"Could not detect GPUs: {e}")
        return []


def patch_transferbench_cmake(repo_path, gpu_list):
    """Patch TransferBench CMakeLists.txt to build only for detected GPUs"""
    cmake_file = repo_path / "plugins" / "tb" / "transferbench" / "CMakeLists.txt"
    
    if not cmake_file.exists():
        log.warning(f"TransferBench CMakeLists.txt not found at {cmake_file}")
        return False
    
    log.info(f"Patching TransferBench for GPU targets: {gpu_list}")
    
    with open(cmake_file, 'r') as f:
        content = f.read()
    
    # Create new GPU list
    gpu_entries = '\n'.join([f"      {gpu}" for gpu in gpu_list])
    new_gpu_section = f"set(DEFAULT_GPUS\n{gpu_entries})"
    
    # Replace the DEFAULT_GPUS section
    pattern = r'set\(DEFAULT_GPUS[^)]+\)'
    
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, new_gpu_section, content, flags=re.DOTALL)
        with open(cmake_file, 'w') as f:
            f.write(content)
        log.info("✓ Patched TransferBench CMakeLists.txt")
        return True
    else:
        log.warning("Could not find DEFAULT_GPUS section to patch")
        return False


def update_submodules(repo_path):
    """Update git submodules (required for TransferBench)"""
    log.info("Updating git submodules...")
    
    # Initialize and sync submodules
    run_cmd("git submodule init", cwd=repo_path, check=False)
    run_cmd("git submodule sync --recursive", cwd=repo_path, check=False)
    run_cmd("git submodule update --init --recursive --force", cwd=repo_path)
    
    log.info("✓ Submodules updated")


def build_rbt(args):
    """Build ROCm Bandwidth Test"""
    
    # Setup paths
    external_dir = THEROCK_DIR / "external"
    repo_path = external_dir / "rocm_bandwidth_test"
    build_dir = repo_path / "build"
    
    # Determine ROCm path
    rocm_dir = Path(args.rocm_path or os.getenv("ROCM_PATH") or os.getenv("ROCM_DIR") or "/opt/rocm")
    rocm_dir = rocm_dir.resolve()
    
    # Install path defaults to ROCm path
    install_dir = Path(args.install_path).resolve() if args.install_path else rocm_dir
    
    # Build type
    build_type = "Debug" if args.debug else "Release"
    
    log.info("=" * 60)
    log.info("ROCm Bandwidth Test - Standalone Builder")
    log.info("=" * 60)
    log.info(f"ROCm Path:    {rocm_dir}")
    log.info(f"Install Path: {install_dir}")
    log.info(f"Build Type:   {build_type}")
    log.info(f"Repository:   {repo_path}")
    log.info("=" * 60)
    
    # Verify ROCm installation
    hipcc = rocm_dir / "bin" / "hipcc"
    if not hipcc.exists():
        log.error(f"hipcc not found at {hipcc}")
        log.error("Please set --rocm-path or ROCM_PATH to a valid ROCm installation")
        sys.exit(1)
    
    # Clone repository if needed
    if not repo_path.exists():
        log.info(f"Cloning RBT repository...")
        external_dir.mkdir(parents=True, exist_ok=True)
        run_cmd(
            "git clone https://github.com/ROCm/rocm_bandwidth_test.git",
            cwd=external_dir
        )
    
    # Update submodules
    update_submodules(repo_path)
    
    # Detect GPUs and patch TransferBench
    if not args.skip_gpu_detect:
        local_gpus = detect_local_gpus(rocm_dir)
        if local_gpus:
            patch_transferbench_cmake(repo_path, local_gpus)
        else:
            log.warning("GPU detection failed, building for default architectures")
    
    # Clean build directory if requested
    if args.clean and build_dir.exists():
        log.info(f"Cleaning build directory...")
        shutil.rmtree(build_dir)
    
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment
    env = {
        "HIP_PLATFORM": "amd",
        "ROCM_PATH": str(rocm_dir),
    }
    
    # Check for toolchain file
    toolchain_file = repo_path / "cmake" / "rocm_clang_toolchain.cmake"
    use_toolchain = toolchain_file.exists()
    
    # Build RPATH settings (from official build script)
    lib_rpath = "$ORIGIN/../lib:$ORIGIN/../lib64:$ORIGIN/../lib/llvm/lib"
    
    # CMake arguments
    cmake_args = [
        f"-DCMAKE_BUILD_TYPE={build_type}",
        "-DAMD_APP_STANDALONE_BUILD_PACKAGE=OFF",
        "-DAMD_APP_ROCM_BUILD_PACKAGE=ON",
        f"-DCMAKE_PREFIX_PATH={rocm_dir}",
        f"-DROCM_PATH={rocm_dir}",
        f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        f'-DCMAKE_INSTALL_RPATH="{lib_rpath}"',
        "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
        "-DCMAKE_SKIP_BUILD_RPATH=OFF",
        # Linker flags for proper RPATH handling
        f'-DCMAKE_EXE_LINKER_FLAGS_INIT="-Wl,--enable-new-dtags,--build-id=sha1"',
        f'-DCMAKE_SHARED_LINKER_FLAGS_INIT="-Wl,--enable-new-dtags,--build-id=sha1"',
    ]
    
    if use_toolchain:
        log.info(f"Using toolchain file: {toolchain_file}")
        cmake_args.append(f"-DCMAKE_TOOLCHAIN_FILE={toolchain_file}")
    
    if args.verbose:
        cmake_args.append("-DCMAKE_VERBOSE_MAKEFILE=ON")
    
    # Run CMake configure
    log.info("Configuring with CMake...")
    cmake_cmd = f"cmake {' '.join(cmake_args)} .."
    run_cmd(cmake_cmd, cwd=build_dir, env=env)
    
    # Build
    num_procs = os.cpu_count() or 8
    log.info(f"Building with {num_procs} parallel jobs...")
    
    build_cmd = f"cmake --build . -j{num_procs}"
    if args.verbose:
        build_cmd += " --verbose"
    run_cmd(build_cmd, cwd=build_dir, env=env)
    
    # Install
    if not args.no_install:
        log.info(f"Installing to {install_dir}...")
        
        # Check if we need sudo for system paths
        need_sudo = False
        try:
            test_file = install_dir / ".write_test"
            test_file.touch()
            test_file.unlink()
        except PermissionError:
            need_sudo = True
        
        install_cmd = "cmake --install ."
        if need_sudo:
            log.info("Using sudo for installation (system path)")
            install_cmd = f"sudo {install_cmd}"
        
        run_cmd(install_cmd, cwd=build_dir, env=env)
    
    # Print summary
    log.info("")
    log.info("=" * 60)
    log.info("✓ Build Complete!")
    log.info("=" * 60)
    log.info(f"Binary:   {install_dir}/bin/rocm-bandwidth-test")
    log.info(f"Libs:     {install_dir}/lib/")
    log.info(f"Plugins:  {install_dir}/lib/rocm_bandwidth_test/plugins/")
    log.info("")
    log.info("Run commands:")
    log.info(f"  {install_dir}/bin/rocm-bandwidth-test plugin --list")
    log.info(f"  {install_dir}/bin/rocm-bandwidth-test run Hello")
    log.info(f"  {install_dir}/bin/rocm-bandwidth-test run tb")
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Build ROCm Bandwidth Test standalone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Build with system ROCm
  %(prog)s --rocm-path ./build          # Build with TheRock artifacts
  %(prog)s --clean --debug              # Clean debug build
  %(prog)s --install-path /tmp/rbt      # Custom install location
"""
    )
    
    parser.add_argument(
        "--rocm-path",
        help="Path to ROCm installation (default: $ROCM_PATH or /opt/rocm)"
    )
    parser.add_argument(
        "--install-path",
        help="Installation path (default: same as rocm-path)"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean build directory before building"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Build debug version instead of release"
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Skip installation step"
    )
    parser.add_argument(
        "--skip-gpu-detect",
        action="store_true",
        help="Skip GPU auto-detection (build for all architectures)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose CMake output"
    )
    
    args = parser.parse_args()
    build_rbt(args)


if __name__ == "__main__":
    main()
