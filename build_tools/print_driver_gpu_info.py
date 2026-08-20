#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Sanity check script for CI runners.

On Linux:
  - run "amd-smi static"
  - run "rocminfo"

On Windows:
  - run "hipInfo.exe"

This script prints only raw command output.
"""

import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
from typing import List, Optional

# Import ASAN helpers from amdgpu_family_matrix
_THIS_DIR = Path(__file__).resolve().parent
_GITHUB_ACTIONS_DIR = _THIS_DIR / "github_actions"
if _GITHUB_ACTIONS_DIR.exists():
    sys.path.insert(0, str(_GITHUB_ACTIONS_DIR))
    from amdgpu_family_matrix import get_asan_lib_path, is_asan_instrumented
else:
    # Fallback if running standalone
    def is_asan_instrumented():
        return os.getenv("BUILD_VARIANT", "") in ("asan", "host-asan")

    def get_asan_lib_path(bin_dir):
        arch = platform.machine()
        clang_path = str(Path(bin_dir).parent / "lib" / "llvm" / "bin" / "clang++")
        asan_lib = f"libclang_rt.asan-{arch}.so"
        cmd = [clang_path, f"-print-file-name={asan_lib}"]
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        resolved = result.stdout.strip()
        if not resolved or resolved == asan_lib or not Path(resolved).is_file():
            raise FileNotFoundError(f"Could not locate ASan runtime '{asan_lib}'")
        return str(Path(resolved).resolve())


def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def run_command(
    args: List[str | Path], cwd: Optional[Path] = None, env: Optional[dict] = None
) -> None:
    args = [str(arg) for arg in args]
    if cwd is None:
        cwd = Path.cwd()

    log(f"++ Exec [{cwd}]$ {shlex.join(args)}")

    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
            stdin=subprocess.DEVNULL,
            env=run_env,
        )
        log(proc.stdout.rstrip())
    except subprocess.CalledProcessError as e:
        # Print the output even on failure so we can diagnose the issue
        if e.stdout:
            log(e.stdout.rstrip())
        raise
    except FileNotFoundError:
        log(f"{args[0]}: command not found")


def run_command_with_search(
    label: str,
    command: str,
    args: List[str],
    extra_command_search_paths: List[Path],
    env: Optional[dict] = None,
) -> None:
    """
    Run a command, searching in extra paths first, then PATH.

    Example:
        run_command_with_search(
            label="amd-smi static",
            command="amd-smi",
            args=["static"],
            extra_command_search_paths=[bin_dir],
        )
    """
    # Try explicit directories first (e.g. THEROCK_DIR/build/bin)
    for base in extra_command_search_paths:
        candidate = base / command
        if candidate.exists():
            log(f"\n=== {label} ===")
            run_command([candidate] + args, env=env)
            return

    # Then fall back to PATH
    resolved = shutil.which(command)
    if resolved:
        log(f"\n=== {label} ===")
        run_command([resolved] + args, env=env)
        return

    # Nothing found
    log(f"\n=== {label} ===")
    log(f"{command}: command not found")


def run_sanity(os_name: str) -> None:
    THIS_SCRIPT_DIR = Path(__file__).resolve().parent
    THEROCK_DIR = THIS_SCRIPT_DIR.parent
    bin_dir = Path(os.getenv("THEROCK_BIN_DIR", THEROCK_DIR / "build" / "bin"))

    log("=== Sanity check: driver / GPU info ===")

    # Set up ASAN runtime preload for instrumented builds (asan or host-asan).
    # Executables that load ASAN-instrumented ROCm libraries need the runtime
    # preloaded, otherwise they abort with "ASan runtime does not come first
    # in initial library list".
    asan_env: Optional[dict] = None
    if os_name.lower() != "windows" and is_asan_instrumented():
        try:
            asan_lib = get_asan_lib_path(bin_dir)
            asan_env = {
                "LD_PRELOAD": asan_lib,
                "ASAN_OPTIONS": "detect_leaks=0",
            }
            log(f"ASAN instrumented build detected, preloading: {asan_lib}")
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            log(f"Warning: Could not resolve ASAN runtime: {e}")

    if os_name.lower() == "windows":
        # Windows: only hipInfo.exe
        run_command_with_search(
            label="hipInfo.exe",
            command="hipInfo.exe",
            args=[],
            extra_command_search_paths=[bin_dir],
        )
    else:
        # Linux: amd-smi static + rocminfo
        run_command_with_search(
            label="amd-smi static",
            command="amd-smi",
            args=["static"],
            extra_command_search_paths=[bin_dir],
            env=asan_env,
        )
        run_command_with_search(
            label="rocminfo",
            command="rocminfo",
            args=[],
            extra_command_search_paths=[bin_dir],
            env=asan_env,
        )
        run_command_with_search(
            label="Kernel version",
            command="uname",
            args=["-r"],
            extra_command_search_paths=[bin_dir],
        )

    log("\n=== End of sanity check ===")


def main(argv: Optional[List[str]] = None) -> int:
    detected = platform.system()
    run_sanity(detected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
