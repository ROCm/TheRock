# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Runs the CPU-only HRR tests from the installed artifact tree.
#
# HRR installs a relocatable ctest tree at share/hrr/test. It finds
# bin/hrr-playback relative to itself, so this script only points ctest at it.
# Only the "cpu-only" label runs: the simulator and GPU suites need a
# simulator or a GPU and are not run from the install tree.

import logging
import os
import shlex
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
THEROCK_DIR = Path(__file__).resolve().parent.parent.parent.parent

if THEROCK_BIN_DIR is None:
    logging.error("env(THEROCK_BIN_DIR) is not set. Set it before running tests.")
    raise SystemExit(1)

HRR_TEST_DIR = Path(THEROCK_BIN_DIR).resolve().parent / "share" / "hrr" / "test"
if not HRR_TEST_DIR.is_dir():
    logging.error(f"HRR test directory not found: {HRR_TEST_DIR}")
    raise SystemExit(1)

cmd = [
    "ctest",
    "--test-dir",
    str(HRR_TEST_DIR),
    "-L",
    "cpu-only",
    "--output-on-failure",
    "--no-tests=error",
]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True)
