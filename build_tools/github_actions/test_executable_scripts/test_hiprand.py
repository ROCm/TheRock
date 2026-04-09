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

import sys
sys.path.append(str(THEROCK_DIR / "build_tools" / "github_actions"))
from github_actions_api import is_host_asan

if is_host_asan():
    print(
        "SKIP: host-ASAN build (THEROCK_SANITIZER=HOST_ASAN) does not produce "
        "GPU math library test binaries. Run with linux-release-asan for device tests."
    )
    sys.exit(0)

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

subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=True,
)
