# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import sys
from pathlib import Path

# Import the ctest retry helper
sys.path.append(str(Path(__file__).resolve().parent))
from ctest_retry_helper import run_ctest_with_retry

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO)

cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/hipRAND",
    "--output-on-failure",
    "--parallel",
    "8",
    "--timeout",
    "60",
]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

exit_code = run_ctest_with_retry(cmd, THEROCK_DIR)
exit(exit_code)
