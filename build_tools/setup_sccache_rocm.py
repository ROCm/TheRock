#!/usr/bin/env python
"""Sets up sccache to wrap ROCm compilers for HIP builds.

This script wraps the ROCm LLVM compilers (clang, clang++) with sccache stubs
to enable caching of HIP device code compilation. This is necessary because
hipcc invokes clang via absolute paths, bypassing CMAKE_*_COMPILER_LAUNCHER.

The approach mirrors what pytorch/pytorch does in their CI:
https://github.com/pytorch/pytorch/blob/main/.ci/docker/common/install_cache.sh

WARNING — Side effects (setup_rocm_sccache):
    This script **modifies the ROCm installation in-place**:
    1. Backs up original compilers (or symlink targets) into a
       ``backup_for_sccache/`` directory alongside the LLVM bin dir.
    2. Replaces clang/clang++ with shell wrapper scripts that invoke
       ``sccache <real-compiler> "$@"``.
    3. You MUST call ``--restore`` (CLI) or ``restore_rocm_compilers()``
       (API) to undo the wrapping. When used via build_prod_wheels.py,
       the try/finally block handles this automatically.

TODO: Once HIP_CLANG_LAUNCHER lands in hipcc (ROCm/llvm-project#72,
ROCm/llvm-project#1490), this wrapper approach can be replaced by
setting the HIP_CLANG_LAUNCHER env var. See also ROCm/ROCm#2817.

Usage:
    # Wrap ROCm compilers with sccache
    python setup_sccache_rocm.py --rocm-path /path/to/rocm

    # Restore original compilers
    python setup_sccache_rocm.py --rocm-path /path/to/rocm --restore

Prerequisites:
    sccache must be installed and available in PATH.
    Install: https://github.com/mozilla/sccache#installation
    For CI, sccache is pre-installed in the manylinux build image:
      https://github.com/ROCm/TheRock/tree/main/dockerfiles
"""

import argparse
import os
import platform
import shutil
import stat
import subprocess
from pathlib import Path

is_windows = platform.system() == "Windows"


def find_sccache() -> Path | None:
    """Find sccache binary in PATH or common locations."""
    sccache_path = shutil.which("sccache")
    if sccache_path:
        return Path(sccache_path)

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


def create_sccache_wrapper(compiler_path: Path, sccache_path: Path) -> None:
    """Create an sccache wrapper for a compiler (Linux only).

    Replaces the compiler (or symlink) with a wrapper script that invokes
    sccache with the resolved absolute path to the real compiler binary.

    Side effects on the filesystem:
      - Creates ``<llvm-bin>/backup_for_sccache/`` directory
      - Moves real binaries (or records symlink targets) into that directory
      - Replaces the compiler at ``compiler_path`` with a shell script

    Use ``restore_compiler()`` to reverse these changes.
    """
    if not compiler_path.exists():
        print(f"  Skipping {compiler_path} (does not exist)")
        return

    compiler_dir = compiler_path.parent
    backup_dir = compiler_dir / "backup_for_sccache"
    backup_dir.mkdir(exist_ok=True)

    backup_path_file = backup_dir / f"{compiler_path.name}.path"

    if backup_path_file.exists():
        print(f"  {compiler_path} already wrapped (backup path file exists)")
        return

    try:
        real_compiler = compiler_path.resolve(strict=True)
        print(f"  Resolved {compiler_path.name} -> {real_compiler}")
    except (OSError, RuntimeError) as e:
        raise RuntimeError(
            f"Failed to resolve compiler path {compiler_path}: {e}"
        ) from e

    if not real_compiler.exists():
        raise RuntimeError(f"Resolved compiler does not exist: {real_compiler}")
    if not os.access(real_compiler, os.X_OK):
        raise RuntimeError(f"Resolved compiler is not executable: {real_compiler}")

    is_symlink = compiler_path.is_symlink()
    original_binary = None

    try:
        if is_symlink:
            original_target = os.readlink(compiler_path)
            backup_path_file.write_text(f"symlink:{original_target}")
        else:
            backup_path_file.write_text(f"binary:{real_compiler}")
            original_binary = backup_dir / compiler_path.name
    except (OSError, PermissionError) as e:
        raise RuntimeError(
            f"Failed to save compiler metadata for {compiler_path}: {e}"
        ) from e

    wrapper_content = f'#!/bin/sh\nexec "{sccache_path}" "{real_compiler}" "$@"\n'

    # For binaries, write wrapper to temp location first to avoid orphaned state
    wrapper_temp = None
    if original_binary is not None:
        wrapper_temp = (
            compiler_path.parent / f".{compiler_path.name}.sccache_wrapper.tmp"
        )
        try:
            wrapper_temp.write_text(wrapper_content)
            wrapper_temp.chmod(
                stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
            )
        except (OSError, PermissionError) as e:
            raise RuntimeError(
                f"Failed to create sccache wrapper for {compiler_path}: {e}"
            ) from e

    if is_symlink:
        compiler_path.unlink()

    try:
        if original_binary is not None:
            shutil.move(compiler_path, original_binary)
            print(f"  Moved binary {compiler_path} -> {original_binary}")
            wrapper_temp.replace(compiler_path)
            print(
                f"  Created sccache wrapper: {compiler_path} -> sccache {real_compiler}"
            )
        else:
            compiler_path.write_text(wrapper_content)
            compiler_path.chmod(
                stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
            )
            print(
                f"  Created sccache wrapper: {compiler_path} -> sccache {real_compiler}"
            )
    except (OSError, PermissionError, shutil.Error) as e:
        if original_binary is not None and original_binary.exists():
            try:
                shutil.move(original_binary, compiler_path)
            except Exception:
                pass
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
    backup_dir = compiler_dir / "backup_for_sccache"
    backup_path_file = backup_dir / f"{compiler_name}.path"
    backup_binary = backup_dir / compiler_name

    if not backup_path_file.exists():
        print(f"  {compiler_path}: no backup path file to restore from")
        return

    path_info = backup_path_file.read_text().strip()

    if compiler_path.exists() or compiler_path.is_symlink():
        compiler_path.unlink()

    if path_info.startswith("symlink:"):
        symlink_target = path_info[len("symlink:") :]
        compiler_path.symlink_to(symlink_target)
        print(f"  Restored symlink {compiler_path} -> {symlink_target}")
    elif path_info.startswith("binary:"):
        if backup_binary.exists():
            shutil.move(backup_binary, compiler_path)
            print(f"  Restored binary {backup_binary} -> {compiler_path}")
        else:
            print(f"  Warning: Original binary not found: {backup_binary}")

    backup_path_file.unlink()

    try:
        backup_dir.rmdir()
    except OSError:
        pass


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


def main():
    parser = argparse.ArgumentParser(
        description="Setup sccache to wrap ROCm compilers for HIP builds"
    )
    parser.add_argument(
        "--rocm-path",
        type=Path,
        required=True,
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

    args = parser.parse_args()

    if args.restore:
        restore_rocm_compilers(args.rocm_path)
        return

    if args.sccache_path:
        sccache_path = args.sccache_path
        if not sccache_path.exists():
            raise RuntimeError(f"Specified sccache not found: {sccache_path}")
    else:
        sccache_path = find_sccache()
        if not sccache_path:
            raise RuntimeError(
                "sccache not found.\n"
                "Install: https://github.com/mozilla/sccache#installation\n"
                "For CI, sccache is pre-installed in the manylinux build image:\n"
                "  https://github.com/ROCm/TheRock/tree/main/dockerfiles"
            )

    print(f"Using sccache: {sccache_path}")

    try:
        result = subprocess.run(
            [str(sccache_path), "--version"], capture_output=True, text=True, check=True
        )
        print(f"sccache version: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"sccache verification failed: {e}") from e

    setup_rocm_sccache(args.rocm_path, sccache_path)


if __name__ == "__main__":
    main()
