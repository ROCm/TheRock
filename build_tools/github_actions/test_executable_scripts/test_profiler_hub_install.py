#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
profiler-hub installation consumption test. Runs the consumer executable
against the assembled profiler-hub artifact tree, proving that tree is
self-sufficient on a machine with no build tree and no compiler.
"""

import logging
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)

SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

# Importing is_asan from amdgpu_family_matrix.py
sys.path.append(str(THEROCK_DIR / "build_tools" / "github_actions"))
from amdgpu_family_matrix import is_asan

OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
if not OUTPUT_ARTIFACTS_DIR:
    raise RuntimeError("OUTPUT_ARTIFACTS_DIR environment variable not set")

ARTIFACTS_DIR = Path(OUTPUT_ARTIFACTS_DIR).resolve()
EXECUTABLE = ARTIFACTS_DIR / "bin" / "test_profiler-hub"

if not EXECUTABLE.is_file():
    logging.error(
        f"consumer executable not found: {EXECUTABLE}. It is built during the "
        "build phase and delivered in the test component of the profiler-hub "
        "artifact; its absence means that component was not fetched or not built."
    )
    raise SystemExit(1)


def get_asan_lib_path():
    clang_path = str(ARTIFACTS_DIR / "lib" / "llvm" / "bin" / "clang++")
    # Which name exists depends on LLVM_ENABLE_PER_TARGET_RUNTIME_DIR.
    asan_libs = ("libclang_rt.asan.so", f"libclang_rt.asan-{platform.machine()}.so")
    for asan_lib in asan_libs:
        cmd = [clang_path, f"-print-file-name={asan_lib}"]
        logging.info(f"++ Exec [{clang_path}]$ {shlex.join(cmd)}")
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        resolved = result.stdout.strip()
        if resolved and resolved != asan_lib and Path(resolved).is_file():
            return str(Path(resolved).resolve())
    raise FileNotFoundError(
        f"Could not locate ASan runtime via {clang_path} (tried: {', '.join(asan_libs)})"
    )


# Popped deliberately: the executable must resolve libprofiler-hub.so via its
# own RUNPATH, and an inherited path would mask a wrong or absent one.
env = os.environ.copy()
env.pop("LD_LIBRARY_PATH", None)
if is_asan():
    asan_lib = get_asan_lib_path()
    existing_preload = env.get("LD_PRELOAD", "")
    env["LD_PRELOAD"] = (
        f"{existing_preload}:{asan_lib}" if existing_preload else asan_lib
    )

logging.info(f"++ Exec [{ARTIFACTS_DIR}]$ {shlex.join([str(EXECUTABLE)])}")
subprocess.run(
    [str(EXECUTABLE)],
    cwd=ARTIFACTS_DIR,
    check=True,
    env=env,
)
