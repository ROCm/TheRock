#!/usr/bin/env python
"""Sets up sccache to wrap ROCm compilers for PyTorch builds.

This script wraps the ROCm LLVM compilers (clang, clang++) with sccache stubs
to enable caching of HIP device code compilation. This is necessary because
hipcc invokes clang via absolute paths, bypassing CMAKE_*_COMPILER_LAUNCHER.

The approach mirrors what pytorch/pytorch does in their CI:
https://github.com/pytorch/pytorch/blob/main/.ci/docker/common/install_cache.sh

Usage:
    # Wrap ROCm compilers with sccache
    python setup_sccache_rocm.py --rocm-path /path/to/rocm

    # Restore original compilers
    python setup_sccache_rocm.py --rocm-path /path/to/rocm --restore

Environment variables for sccache configuration:
    SCCACHE_BUCKET: S3 bucket for remote caching
    SCCACHE_REGION: S3 region
    SCCACHE_DIR: Local cache directory (if not using remote)
"""

import argparse
import os
import platform
import shutil
import stat
import subprocess
import sys
from pathlib import Path

is_windows = platform.system() == "Windows"


def find_sccache() -> Path | None:
    """Find sccache binary in PATH or common locations."""
    # Check PATH first
    sccache_path = shutil.which("sccache")
    if sccache_path:
        return Path(sccache_path)

    # Check common locations
    common_paths = [
        Path("/usr/local/bin/sccache"),
        Path("/opt/cache/bin/sccache"),
        Path.home() / ".cargo" / "bin" / "sccache",
    ]
    if is_windows:
        common_paths.extend(
            [
                Path("C:/ProgramData/chocolatey/bin/sccache.exe"),
                Path.home() / ".cargo" / "bin" / "sccache.exe",
            ]
        )

    for path in common_paths:
        if path.exists():
            return path

    return None


def install_sccache() -> Path:
    """Install sccache if not available."""
    sccache_path = find_sccache()
    if sccache_path:
        print(f"Found sccache at: {sccache_path}")
        return sccache_path

    print("sccache not found, attempting to install...")

    if is_windows:
        # Try cargo install
        try:
            subprocess.check_call(["cargo", "install", "sccache"])
            sccache_path = Path.home() / ".cargo" / "bin" / "sccache.exe"
            if sccache_path.exists():
                return sccache_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        raise RuntimeError(
            "Could not install sccache. Please install it manually:\n"
            "  choco install sccache\n"
            "  or: cargo install sccache"
        )
    else:
        # Try pip install (sccache is available on PyPI)
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sccache"])
            sccache_path = find_sccache()
            if sccache_path:
                return sccache_path
        except subprocess.CalledProcessError:
            pass

        # Try cargo install as fallback
        try:
            subprocess.check_call(["cargo", "install", "sccache"])
            sccache_path = Path.home() / ".cargo" / "bin" / "sccache"
            if sccache_path.exists():
                return sccache_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        raise RuntimeError(
            "Could not install sccache. Please install it manually:\n"
            "  pip install sccache\n"
            "  or: cargo install sccache"
        )


def create_sccache_wrapper(compiler_path: Path, sccache_path: Path) -> None:
    """Create an sccache wrapper for a compiler (Linux only).

    This replaces the compiler (or symlink) with a wrapper script that invokes
    sccache with the resolved absolute path to the real compiler binary.
    """
    if not compiler_path.exists():
        print(f"  Skipping {compiler_path} (does not exist)")
        return

    compiler_dir = compiler_path.parent
    original_dir = compiler_dir / "original"

    # Create original directory for metadata
    original_dir.mkdir(exist_ok=True)

    # Store the original target path in a file for restoration
    original_path_file = original_dir / f"{compiler_path.name}.path"

    # Check if already wrapped
    if original_path_file.exists():
        print(f"  {compiler_path} already wrapped (path file exists)")
        return

    # Resolve the compiler to its real absolute path (following all symlinks)
    try:
        real_compiler = compiler_path.resolve(strict=True)
        print(f"  Resolved {compiler_path.name} -> {real_compiler}")
    except (OSError, RuntimeError) as e:
        raise RuntimeError(
            f"Failed to resolve compiler path {compiler_path}: {e}"
        ) from e

    # Verify the resolved compiler exists and is executable
    if not real_compiler.exists():
        raise RuntimeError(f"Resolved compiler does not exist: {real_compiler}")
    if not os.access(real_compiler, os.X_OK):
        raise RuntimeError(f"Resolved compiler is not executable: {real_compiler}")

    # Save the original symlink target (if symlink) or path for restoration
    is_symlink = compiler_path.is_symlink()
    original_binary = None
    
    try:
        if is_symlink:
            original_target = os.readlink(compiler_path)
            original_path_file.write_text(f"symlink:{original_target}")
        else:
            # Save metadata for binary (don't move yet - move after wrapper is ready)
            original_path_file.write_text(f"binary:{real_compiler}")
            original_binary = original_dir / compiler_path.name
    except (OSError, PermissionError) as e:
        raise RuntimeError(f"Failed to save compiler metadata for {compiler_path}: {e}") from e

    # Prepare wrapper content
    wrapper_content = f'#!/bin/sh\nexec "{sccache_path}" "{real_compiler}" "$@"\n'
    
    # For binaries, create wrapper at temp location first to verify we can write it
    # before moving the binary (avoids orphaned state if wrapper creation fails)
    wrapper_temp = None
    if original_binary is not None:
        wrapper_temp = compiler_path.parent / f".{compiler_path.name}.sccache_wrapper.tmp"
        try:
            wrapper_temp.write_text(wrapper_content)
            wrapper_temp.chmod(
                stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
            )
        except (OSError, PermissionError) as e:
            raise RuntimeError(
                f"Failed to create sccache wrapper for {compiler_path}: {e}"
            ) from e

    # Remove existing symlink if present
    if is_symlink:
        compiler_path.unlink()

    # For binaries: move binary to original/, then move wrapper to final location
    # For symlinks: create wrapper directly (symlink already removed above)
    try:
        if original_binary is not None:
            # Move binary to original/ for safekeeping
            shutil.move(compiler_path, original_binary)
            print(f"  Moved binary {compiler_path} -> {original_binary}")
            # Move wrapper from temp to final location
            wrapper_temp.replace(compiler_path)
            print(f"  Created sccache wrapper: {compiler_path} -> sccache {real_compiler}")
        else:
            # For symlinks, create wrapper directly
            compiler_path.write_text(wrapper_content)
            compiler_path.chmod(
                stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
            )
            print(f"  Created sccache wrapper: {compiler_path} -> sccache {real_compiler}")
    except (OSError, PermissionError, shutil.Error) as e:
        # Cleanup on failure: restore binary if it was moved
        if original_binary is not None and original_binary.exists():
            try:
                shutil.move(original_binary, compiler_path)
            except Exception:
                pass  # Best effort restore
        # Clean up temp wrapper if it exists
        if wrapper_temp is not None and wrapper_temp.exists():
            try:
                wrapper_temp.unlink()
            except Exception:
                pass
        raise RuntimeError(
            f"Failed to create sccache wrapper for {compiler_path}: {e}"
        ) from e


def restore_compiler(compiler_path: Path) -> None:
    """Restore original compiler by removing sccache wrapper (Linux only)."""
    compiler_name = compiler_path.name
    compiler_dir = compiler_path.parent
    original_dir = compiler_dir / "original"
    original_path_file = original_dir / f"{compiler_name}.path"
    original_binary = original_dir / compiler_name

    if not original_path_file.exists():
        print(f"  {compiler_path}: no path file to restore from")
        return

    # Read the original path info
    path_info = original_path_file.read_text().strip()

    # Remove the wrapper
    if compiler_path.exists() or compiler_path.is_symlink():
        compiler_path.unlink()

    # Restore based on type
    if path_info.startswith("symlink:"):
        # Restore the original symlink
        symlink_target = path_info[8:]  # Remove "symlink:" prefix
        compiler_path.symlink_to(symlink_target)
        print(f"  Restored symlink {compiler_path} -> {symlink_target}")
    elif path_info.startswith("binary:"):
        # Restore the moved binary
        if original_binary.exists():
            shutil.move(original_binary, compiler_path)
            print(f"  Restored binary {original_binary} -> {compiler_path}")
        else:
            print(f"  Warning: Original binary not found: {original_binary}")

    # Clean up path file
    original_path_file.unlink()

    # Clean up original dir if empty
    try:
        original_dir.rmdir()
    except OSError:
        pass  # Not empty


def _find_rocm_llvm_bin(rocm_path: Path) -> Path | None:
    """Find the ROCm LLVM bin directory."""
    candidates = [
        rocm_path / "lib" / "llvm" / "bin",
        rocm_path / "llvm" / "bin",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def setup_rocm_sccache(rocm_path: Path, sccache_path: Path) -> None:
    """Wrap ROCm compilers with sccache (Linux only).

    On Windows, compiler wrapping is skipped because hipcc calls clang.exe
    directly and shell script wrappers won't intercept these calls.
    Windows builds rely on CMAKE_C/CXX_COMPILER_LAUNCHER for host code caching.
    Note: CMAKE_HIP_COMPILER_LAUNCHER is not supported by sccache.
    """
    if is_windows:
        print("Skipping ROCm compiler wrapping on Windows (using CMAKE launchers)")
        return

    llvm_bin = _find_rocm_llvm_bin(rocm_path)
    if not llvm_bin:
        raise RuntimeError(
            f"Could not find ROCm LLVM bin directory. Tried:\n"
            f"  {rocm_path / 'lib' / 'llvm' / 'bin'}\n"
            f"  {rocm_path / 'llvm' / 'bin'}"
        )

    print(f"Setting up sccache wrappers in {llvm_bin}")
    for compiler in ["clang", "clang++"]:
        create_sccache_wrapper(llvm_bin / compiler, sccache_path)
    print("ROCm compiler sccache wrapping complete.")


def restore_rocm_compilers(rocm_path: Path) -> None:
    """Restore original ROCm compilers (Linux only)."""
    if is_windows:
        return

    llvm_bin = _find_rocm_llvm_bin(rocm_path)
    if not llvm_bin:
        print("Warning: Could not find ROCm LLVM bin directory")
        return

    print(f"Restoring original compilers in {llvm_bin}")
    for compiler in ["clang", "clang++"]:
        restore_compiler(llvm_bin / compiler)
    print("ROCm compiler restoration complete.")


def parse_sccache_stats(stats_output: str) -> dict:
    """Parse sccache --show-stats output for metrics.

    Returns a dictionary with keys:
        - compile_requests: Total compilation requests
        - cache_hits: Number of cache hits
        - cache_misses: Number of cache misses
        - hit_rate: Cache hit percentage
        - cache_errors: Number of cache errors
    """
    metrics = {}

    def extract_int(line: str) -> int | None:
        try:
            return int(line.split()[-1])
        except (ValueError, IndexError):
            return None

    for line in stats_output.splitlines():
        line = line.strip()
        if "Compile requests" in line and "compile_requests" not in metrics:
            if (val := extract_int(line)) is not None:
                metrics["compile_requests"] = val
        elif "Cache hits" in line and "(Rust)" not in line:
            if (val := extract_int(line)) is not None:
                metrics["cache_hits"] = val
        elif "Cache misses" in line:
            if (val := extract_int(line)) is not None:
                metrics["cache_misses"] = val
        elif "Cache errors" in line:
            if (val := extract_int(line)) is not None:
                metrics["cache_errors"] = val

    # Calculate hit rate
    if "compile_requests" in metrics and "cache_hits" in metrics:
        total = metrics["compile_requests"]
        metrics["hit_rate"] = (
            (metrics["cache_hits"] / total * 100.0) if total > 0 else 0.0
        )

    return metrics


def print_sccache_stats():
    """Print sccache statistics with cache hit rate analysis."""
    sccache_path = find_sccache()
    if sccache_path:
        try:
            result = subprocess.run(
                [str(sccache_path), "--show-stats"],
                capture_output=True,
                text=True,
                check=False,
            )
            stats_output = result.stdout
            print(stats_output)

            # Parse and display metrics
            metrics = parse_sccache_stats(stats_output)
            if metrics:
                print("\n=== sccache Cache Performance ===")
                if "compile_requests" in metrics:
                    print(f"Total Compile Requests: {metrics['compile_requests']}")
                if "cache_hits" in metrics:
                    print(f"Cache Hits: {metrics['cache_hits']}")
                if "cache_misses" in metrics:
                    print(f"Cache Misses: {metrics['cache_misses']}")
                if "hit_rate" in metrics:
                    print(f"Cache Hit Rate: {metrics['hit_rate']:.1f}%")
                if "cache_errors" in metrics and metrics["cache_errors"] > 0:
                    print(f"⚠️  Cache Errors: {metrics['cache_errors']}")
        except Exception as e:
            print(f"Could not get sccache stats: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Setup sccache to wrap ROCm compilers for PyTorch builds"
    )
    parser.add_argument(
        "--rocm-path",
        type=Path,
        help="Path to ROCm installation (e.g., from `python -m rocm_sdk path --root`)",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore original compilers (remove sccache wrappers)",
    )
    parser.add_argument(
        "--sccache-path",
        type=Path,
        help="Path to sccache binary (auto-detected if not specified)",
    )
    parser.add_argument(
        "--show-stats", action="store_true", help="Show sccache statistics"
    )

    args = parser.parse_args()

    if args.show_stats:
        print_sccache_stats()
        return

    # rocm-path is required for setup and restore operations
    if not args.rocm_path:
        parser.error("--rocm-path is required for setup/restore operations")

    if args.restore:
        restore_rocm_compilers(args.rocm_path)
        return

    # Find or install sccache
    if args.sccache_path:
        sccache_path = args.sccache_path
        if not sccache_path.exists():
            raise RuntimeError(f"Specified sccache not found: {sccache_path}")
    else:
        sccache_path = install_sccache()

    print(f"Using sccache: {sccache_path}")

    # Verify sccache works
    try:
        result = subprocess.run(
            [str(sccache_path), "--version"], capture_output=True, text=True, check=True
        )
        print(f"sccache version: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"sccache verification failed: {e}")

    # Setup wrappers
    setup_rocm_sccache(args.rocm_path, sccache_path)


if __name__ == "__main__":
    main()
