import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO)

# If smoke tests are enabled, we run smoke tests only.
# Otherwise, we run the normal test suite
test_type = os.getenv("TEST_TYPE", "full")

test_subdir = "regression"
if test_type == "smoke":
    test_subdir = "smoke"

cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/rocwmma/{test_subdir}",
    "--output-on-failure",
    "--parallel",
    "8",
    "--timeout",
    "300",
]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=True,
)
