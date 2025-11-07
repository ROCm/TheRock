import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

cmd_lit = ["pip", "install", "lit"]

environ_vars = os.environ.copy()
subprocess.run(cmd_lit, cwd=THEROCK_DIR, check=True, env=environ_vars)

print(THEROCK_DIR, THEROCK_BIN_DIR)

cmd = [
    "cd",
    f"{THEROCK_DIR}/build/math-libs/libhipcxx/build/",
]
LIBHIPCXX_BUILD_DIR=f"{THEROCK_DIR}/build/math-libs/libhipcxx/build/"
try:
    os.chdir(LIBHIPCXX_BUILD_DIR)
    print(f"Changed working directory to: {os.getcwd()}")
except FileNotFoundError:
    print(f"Error: Directory '{LIBHIPCXX_BUILD_DIR}' does not exist.")

cmd = [
    "bash",
    f"{THEROCK_DIR}/math-libs/libhipcxx/utils/amd/linux/perform_tests.bash",
    "--libhipcxx-lit-site-config",
    f"{LIBHIPCXX_BUILD_DIR}/test/lit.site.cfg",
    "--skip-libcxx-tests",
]

# If smoke tests are enabled, we run smoke tests only.
# Otherwise, we run the normal test suite
environ_vars = os.environ.copy()
test_type = os.getenv("TEST_TYPE", "full")
if test_type == "smoke":
    environ_vars["GTEST_FILTER"] = ":".join(SMOKE_TESTS)

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)
