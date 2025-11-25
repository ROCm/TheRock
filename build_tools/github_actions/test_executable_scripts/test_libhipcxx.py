import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

environ_vars = os.environ.copy()

print(THEROCK_DIR, THEROCK_BIN_DIR, OUTPUT_ARTIFACTS_DIR)

LIBHIPCXX_BUILD_DIR = f"{OUTPUT_ARTIFACTS_DIR}/libhipcxx"

try:
    os.chdir(LIBHIPCXX_BUILD_DIR)
    os.mkdir("build")
    os.chdir("build")
    print(f"Changed working directory to: {os.getcwd()}")
except FileNotFoundError:
    print(f"Error: Directory '{LIBHIPCXX_BUILD_DIR}' does not exist.")



cmd = [
    "cmake",
    ".."
]

subprocess.run(cmd, check=True, env=environ_vars)


# If smoke tests are enabled, we run smoke tests only.
# Otherwise, we run the normal test suite
# environ_vars = os.environ.copy()
# test_type = os.getenv("TEST_TYPE", "full")
# if test_type == "smoke":
#     environ_vars["GTEST_FILTER"] = ":".join(SMOKE_TESTS)
cmd = [
    "make",
    "check-hipcxx"
]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

subprocess.run(cmd, check=True, env=environ_vars)
