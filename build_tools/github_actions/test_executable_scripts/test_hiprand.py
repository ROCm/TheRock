# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO)

# When launched under the rocjitsu simulator, simulator_runner.py sets these
# opt-in knobs. They are unset on the real-GPU lane, where behavior below is
# unchanged. (hipRAND has no --repeat to suppress, so SIMULATOR_NO_RETRY is
# not relevant here.)
include_regex = os.getenv("SIMULATOR_CTEST_INCLUDE_REGEX")
test_timeout = os.getenv("CTEST_TEST_TIMEOUT")

cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/hipRAND",
    "--output-on-failure",
]
# Narrow which ctest binaries run so the simulator does not walk the whole suite.
if include_regex:
    cmd += ["-R", include_regex]
# Bound each test so a stall fails fast instead of eating the step budget.
if test_timeout:
    cmd += ["--timeout", test_timeout]

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=True,
)
