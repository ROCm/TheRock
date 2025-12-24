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

  # Build with minimal submodules (faster, skips full boost clone)
  python build_tools/build_rbt_standalone.py --minimal-submodules

  # Clean build
  python build_tools/build_rbt_standalone.py --clean

  # Debug build
  python build_tools/build_rbt_standalone.py --debug

Options:
  --rocm-path PATH        Path to ROCm installation (default: $ROCM_PATH or /opt/rocm)
  --install-path PATH     Install location (default: same as rocm-path)
  --minimal-submodules    Use minimal submodule cloning (skips full boost clone,
                          ~10x faster). Fully self-contained build.
  --clean                 Clean build directory before building
  --debug                 Build debug version instead of release
  --no-install            Skip installation step
  --verbose               Show verbose CMake output
  --help                  Show this help message
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


def update_submodules(repo_path, minimal_submodules=False):
    """Update git submodules (required for TransferBench)
    
    Args:
        repo_path: Path to the rocm_bandwidth_test repository
        minimal_submodules: If True, only clone essential submodules with minimal
                           boost components (skips full 180+ boost submodule tree)
    """
    log.info("Updating git submodules...")
    
    if minimal_submodules:
        # Only update essential submodules when using system deps
        # Skip: CLI11, json, jthread, Catch2 (use system packages or not needed)
        # Keep: fmt, spdlog (version requirements too strict for Ubuntu packages)
        # Keep: boost (Ubuntu's boost doesn't have CMake CONFIG files)
        essential_submodules = [
            "deps/external/TransferBench",
            "deps/3rd_party/fmt",      # Required >= 9.1.0, Ubuntu has 8.1.1
            "deps/3rd_party/spdlog",   # Required >= 1.15.0, Ubuntu has 1.9.2
            "deps/3rd_party/Catch2",   # Required >= 3.5.1, Ubuntu has 2.x
            "deps/3rd_party/jthread",  # For C++ < 20 compatibility (submodule includes wrapper)
            "deps/3rd_party/CLI11",    # Small repo, clone for self-contained build
            "deps/3rd_party/json",     # Small repo, clone for self-contained build
        ]
        log.info("Using minimal submodule cloning - only essential components")
        for submodule in essential_submodules:
            # Initialize and update only this specific submodule
            run_cmd(f"git submodule init {submodule}", cwd=repo_path, check=False)
            run_cmd(f"git submodule update --init --recursive --force {submodule}", 
                   cwd=repo_path, check=False)
        
        # For boost, we need the submodule but Ubuntu's boost doesn't have CMake CONFIG
        # files. Clone only the needed boost component submodules (not the full monorepo)
        log.info("Initializing boost submodule (minimal components for stacktrace)...")
        run_cmd("git submodule init deps/3rd_party/boost", cwd=repo_path, check=False)
        # First checkout the boost root (non-recursive to avoid 180+ submodule downloads)
        run_cmd("git submodule update --init deps/3rd_party/boost", cwd=repo_path, check=False)
        # Now init only the components needed for stacktrace
        boost_components = [
            "config", "core", "assert", "static_assert", "type_traits", 
            "predef", "array", "container_hash", "describe", "throw_exception",
            "mp11", "winapi", "stacktrace"
        ]
        boost_base = repo_path / "deps" / "3rd_party" / "boost"
        for comp in boost_components:
            run_cmd(f"git submodule update --init libs/{comp}", cwd=boost_base, check=False)
    else:
        # Initialize and update all submodules
        run_cmd("git submodule init", cwd=repo_path, check=False)
        run_cmd("git submodule sync --recursive", cwd=repo_path, check=False)
        run_cmd("git submodule update --init --recursive --force", cwd=repo_path)
        
        # Fix jthread submodule - the upstream josuttis/jthread repo doesn't have a
        # CMakeLists.txt but the build system's verify_dependency_support() requires one.
        # Create a stub CMakeLists.txt to satisfy the check.
        jthread_submodule = repo_path / "deps" / "3rd_party" / "jthread" / "jthread"
        jthread_cmake = jthread_submodule / "CMakeLists.txt"
        if jthread_submodule.exists() and not jthread_cmake.exists():
            log.info("Patching jthread submodule with stub CMakeLists.txt...")
            jthread_cmake.write_text(
                "# Stub CMakeLists.txt for jthread submodule\n"
                "# The actual build is handled by the parent jthread/CMakeLists.txt wrapper\n"
                "cmake_minimum_required(VERSION 3.16)\n"
                "project(jthread_submodule)\n"
            )
            log.info("✓ jthread submodule patched")
    
    log.info("✓ Submodules updated")


def patch_rbt_cmake_issues(repo_path):
    """Fix known issues in RBT CMake files"""
    
    # Fix typo in main/cmdline/CMakeLists.txt: "QUIT" should be "QUIET"
    cmdline_cmake = repo_path / "main" / "cmdline" / "CMakeLists.txt"
    if cmdline_cmake.exists():
        content = cmdline_cmake.read_text()
        if "CONFIG QUIT)" in content:
            log.info("Patching CLI11 cmake typo (QUIT -> QUIET)...")
            content = content.replace("CONFIG QUIT)", "CONFIG QUIET)")
            cmdline_cmake.write_text(content)
            log.info("✓ CLI11 cmake typo fixed")


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
    log.info(f"Minimal Clone: {'Yes' if args.minimal_submodules else 'No (full recursive clone)'}")
    log.info("=" * 60)
    
    if args.minimal_submodules:
        log.info("Using minimal submodule cloning (skips full boost clone).")
        log.info("Cloning: TransferBench, fmt, spdlog, Catch2, CLI11, json, jthread")
        log.info("Boost: Only 13 minimal components (vs 180+ in full clone)")
    
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
    update_submodules(repo_path, minimal_submodules=args.minimal_submodules)
    
    # Fix known CMake issues in RBT repo
    patch_rbt_cmake_issues(repo_path)
    
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
    
    # Set environment variables globally so they're inherited by all child processes
    # (including TransferBench's build_libamd_tb.sh script)
    os.environ["HIP_PLATFORM"] = "amd"
    os.environ["ROCM_PATH"] = str(rocm_dir)
    os.environ["HIP_PATH"] = str(rocm_dir)
    os.environ["HIP_CLANG_PATH"] = str(rocm_dir / "lib" / "llvm" / "bin")
    
    # For TheRock layout, device libraries are under lib/llvm/amdgcn/bitcode
    # instead of the standard amdgcn/bitcode
    device_lib_path = rocm_dir / "lib" / "llvm" / "amdgcn" / "bitcode"
    if device_lib_path.exists():
        os.environ["HIP_DEVICE_LIB_PATH"] = str(device_lib_path)
        log.info(f"Using TheRock device library path: {device_lib_path}")
    else:
        # Try standard ROCm layout
        std_device_lib = rocm_dir / "amdgcn" / "bitcode"
        if std_device_lib.exists():
            os.environ["HIP_DEVICE_LIB_PATH"] = str(std_device_lib)
    
    # Add ROCm bin to PATH so hipcc can find clang
    rocm_bin = str(rocm_dir / "bin")
    llvm_bin = str(rocm_dir / "lib" / "llvm" / "bin")
    current_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{rocm_bin}:{llvm_bin}:{current_path}"
    
    # Set library paths
    lib_paths = [
        str(rocm_dir / "lib"),
        str(rocm_dir / "lib64"),
        str(rocm_dir / "lib" / "llvm" / "lib"),
    ]
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if current_ld:
        lib_paths.append(current_ld)
    os.environ["LD_LIBRARY_PATH"] = ":".join(lib_paths)
    
    log.info(f"Environment:")
    log.info(f"  ROCM_PATH={os.environ['ROCM_PATH']}")
    log.info(f"  HIP_PATH={os.environ['HIP_PATH']}")
    log.info(f"  HIP_CLANG_PATH={os.environ['HIP_CLANG_PATH']}")
    if "HIP_DEVICE_LIB_PATH" in os.environ:
        log.info(f"  HIP_DEVICE_LIB_PATH={os.environ['HIP_DEVICE_LIB_PATH']}")
    
    env = None  # Use global environment
    
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
        # Disable tests (Catch2 submodule not needed)
        "-DAMD_APP_BUILD_TESTS=OFF",
    ]
    
    # Use minimal submodule cloning instead of full recursive clone
    if args.minimal_submodules:
        log.info("Using minimal submodule cloning (skipping full boost clone)")
        # Clone only essential submodules:
        # - fmt, spdlog, Catch2, CLI11, json, jthread (small repos, quick to clone)
        # - boost (only 13 minimal components instead of 180+)
        # This avoids the multi-minute boost submodule recursion
        # No external system package dependencies required!
        cmake_args.extend([
            # Use submodules for all (fully self-contained build)
            "-DUSE_LOCAL_CATCH2=OFF",
            "-DUSE_LOCAL_BOOST=OFF",
            "-DUSE_LOCAL_BOOST_STACKTRACE=OFF",
            # Don't set USE_LOCAL_JTHREAD - let cmake auto-detect C++20 and skip it
        ])
    
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
  %(prog)s --minimal-submodules         # Fast build (minimal clone)
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
        "--minimal-submodules",
        action="store_true",
        dest="minimal_submodules",
        help="Use minimal submodule cloning (skips full boost clone, "
             "~10x faster). Fully self-contained build."
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
