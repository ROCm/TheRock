import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO)

# Determine test filter and timeout based on TEST_TYPE environment variable
environ_vars = os.environ.copy()
test_type = os.getenv("TEST_TYPE", "full")

if test_type == "smoke":
    # Exclude tests that start with "Full"
    environ_vars["GTEST_FILTER"] = "-Full*"
    timeout = "300"  # 5 minutes max for Smoke tests
    logging.info("Running smoke tests (excluding Full* tests)")
else:
    timeout = "900"  # 15 minutes max for Full tests
    logging.info("Running full test suite")

cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/miopen_legacy_plugin",
    "--output-on-failure",
    "--parallel",
    "8",
    "--timeout",
    timeout,
]

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=True,
    env=environ_vars,
)
