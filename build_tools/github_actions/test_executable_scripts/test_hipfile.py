# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Runs hipFile's unit tests from the installed/packaged artifact tree.
#
# hipFile installs a relocatable ctest tree at share/hipfile/test (a top-level
# CTestTestfile.cmake plus script/hipfile_discover.cmake). Discovery and the
# runtime library path are handled inside that tree relative to its own
# location, so this script only needs to point ctest at it -- no LD_LIBRARY_PATH
# setup is required here.
#
# Only the "unit" label is run: the "system" tests need a real GPU and the
# "stress" tests are gdb-wrapped concurrency testers, both excluded from the
# packaged unit suite.

import logging
import shlex
import subprocess
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO)

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

if THEROCK_BIN_DIR is None:
    logging.error("env(THEROCK_BIN_DIR) is not set. Set it before running tests.")
    raise SystemExit(1)

# THEROCK_BIN_DIR is <install>/bin; the relocatable test tree is alongside it.
HIPFILE_TEST_DIR = Path(THEROCK_BIN_DIR).resolve().parent / "share" / "hipfile" / "test"

if not HIPFILE_TEST_DIR.is_dir():
    logging.error(f"hipFile test directory not found: {HIPFILE_TEST_DIR}")
    raise SystemExit(1)

cmd = [
    "ctest",
    "--test-dir",
    str(HIPFILE_TEST_DIR),
    "-L",
    "unit",
    "--output-on-failure",
    "--no-tests=error",
]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=True,
)
