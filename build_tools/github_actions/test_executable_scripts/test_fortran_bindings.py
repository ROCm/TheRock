# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
TEST_COMPONENT = os.getenv("TEST_COMPONENT")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO)

# Each test job is named after a binding subproject, so that consumer-based test
# selection picks it when the library it binds changes. A binding installs its
# CTestTestfile.cmake in bin/<binding>-fortran/.
TEST_DIRS = {
    "runtime-fortran": ["hip-fortran", "roctx-fortran"],
    **{
        f"{lib}-fortran": [f"{lib}-fortran"]
        for lib in (
            "rocblas",
            "hipblas",
            "rocsolver",
            "hipsolver",
            "rocsparse",
            "hipsparse",
            "rocfft",
            "hipfft",
            "rocrand",
            "hiprand",
        )
    },
}

if TEST_COMPONENT not in TEST_DIRS:
    logging.error(f"Unknown Fortran binding test component: {TEST_COMPONENT}")
    sys.exit(1)

test_dirs = [Path(THEROCK_BIN_DIR) / d for d in TEST_DIRS[TEST_COMPONENT]]
test_dirs = [d for d in test_dirs if (d / "CTestTestfile.cmake").is_file()]
if not test_dirs:
    # Not an error yet: the monorepo pins do not contain the bindings.
    print(
        f"::warning::No {TEST_COMPONENT} tests are installed in {THEROCK_BIN_DIR}; "
        "nothing was run.",
        flush=True,
    )
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
