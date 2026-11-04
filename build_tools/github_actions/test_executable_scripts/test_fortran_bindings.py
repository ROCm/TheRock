# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO)

# Each binding installs its CTestTestfile.cmake in bin/<binding>-fortran/.
BINDINGS = ["hip-fortran", "roctx-fortran"]

test_dirs = [Path(THEROCK_BIN_DIR) / b for b in BINDINGS]
test_dirs = [d for d in test_dirs if (d / "CTestTestfile.cmake").is_file()]
if not test_dirs:
    logging.info("No Fortran binding tests are installed; nothing to run.")
    sys.exit(0)

failed = []
for test_dir in test_dirs:
    cmd = ["ctest", "--test-dir", str(test_dir), "--output-on-failure"]
    logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
    if subprocess.run(cmd, cwd=THEROCK_DIR).returncode != 0:
        failed.append(test_dir.name)

if failed:
    logging.error(f"Failed Fortran binding tests: {', '.join(failed)}")
    sys.exit(1)
