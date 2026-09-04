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
import shlex
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)

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

# Popped deliberately: the executable must resolve libprofiler-hub.so via its
# own RUNPATH, and an inherited path would mask a wrong or absent one.
env = os.environ.copy()
env.pop("LD_LIBRARY_PATH", None)

logging.info(f"++ Exec [{ARTIFACTS_DIR}]$ {shlex.join([str(EXECUTABLE)])}")
subprocess.run(
    [str(EXECUTABLE)],
    cwd=ARTIFACTS_DIR,
    check=True,
    env=env,
)
