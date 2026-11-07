# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Test script for the amd-dbgapi unit tests.

The test binaries are installed under tests/rocdbgapi in the unified ROCm tree.
They are plain GoogleTest executables, so each one is run directly.
"""

import logging
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
if not THEROCK_BIN_DIR:
    raise RuntimeError("THEROCK_BIN_DIR environment variable is not set")

logging.basicConfig(level=logging.INFO, format="%(message)s")

test_dir = Path(THEROCK_BIN_DIR).resolve().parent / "tests" / "rocdbgapi"

is_windows = platform.system() == "Windows"
tests = sorted(
    p
    for p in test_dir.iterdir()
    if p.is_file() and (p.suffix == ".exe" if is_windows else os.access(p, os.X_OK))
)
if not tests:
    logging.error(f"No test executables found in {test_dir}")
    sys.exit(1)

failed = []
for test in tests:
    cmd = [str(test)]
    logging.info(f"++ Exec [{test_dir}]$ {shlex.join(cmd)}")
    if subprocess.run(cmd, cwd=test_dir).returncode != 0:
        failed.append(test.name)

if failed:
    logging.error(f"Failed tests: {', '.join(failed)}")
    sys.exit(1)
