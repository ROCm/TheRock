# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = Path(os.getenv("THEROCK_BIN_DIR")).resolve()
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO)

fusilli_test_dir = THEROCK_BIN_DIR / "fusilli_plugin_test_infra"

# Build the ctest command
cmd = [
    "ctest",
    "--test-dir",
    str(fusilli_test_dir),
    "--output-on-failure",
    "--parallel",
    "8",
    "--timeout",
    "600",
]

# Set up environment variables
environ_vars = os.environ.copy()

# Determine test filter based on TEST_TYPE environment variable
test_type = os.getenv("TEST_TYPE", "full")
if test_type == "quick":
    # Exclude tests that start with "Full" during quick tests
    environ_vars["GTEST_FILTER"] = "-Full*"

if not fusilli_test_dir.exists():
    logging.info(
        "Fusilli test artifacts not found at %s. Treating this as a pass because "
        "Fusilli is not built in all currently active CI pipelines.",
        fusilli_test_dir,
    )
    raise SystemExit(0)

# As a sanity check, verify libIREECompiler.so is available in the build artifacts.
# TODO: check for .dll on windows
iree_compiler_lib = THEROCK_BIN_DIR.parent / "lib" / "libIREECompiler.so"
if not iree_compiler_lib.exists():
    raise RuntimeError(
        f"libIREECompiler.so not found at {iree_compiler_lib}. "
        "Ensure THEROCK_ENABLE_IREE_COMPILER is ON and iree-compiler is built."
    )
logging.info(f"Verified libIREECompiler.so available at: {iree_compiler_lib}")

# Add THEROCK_BIN_DIR to PATH for rocm_agent_enumerator
environ_vars["PATH"] = f"{THEROCK_BIN_DIR}:{environ_vars['PATH']}"

# Run the tests
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
if test_type == "quick":
    logging.info("   TEST_TYPE=quick: Excluding Full* tests via GTEST_FILTER")
subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=True,
    env=environ_vars,
)
