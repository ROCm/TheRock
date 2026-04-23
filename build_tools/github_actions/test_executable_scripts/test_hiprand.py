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

# Allow external consumers (e.g. FFM runners with tighter resource budgets) to
# override ctest timeout and parallelism without modifying this script. Defaults
# preserve the existing hardcoded values so TheRock's own CI is unaffected.
ctest_timeout = int(os.getenv("CTEST_TIMEOUT_OVERRIDE", "60"))
ctest_parallel = int(os.getenv("CTEST_PARALLEL_OVERRIDE", "8"))

cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/hipRAND",
    "--output-on-failure",
    "--parallel",
    str(ctest_parallel),
    "--timeout",
    str(ctest_timeout),
]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=True,
)
