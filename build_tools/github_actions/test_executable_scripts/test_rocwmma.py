import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")
platform = os.getenv("RUNNER_OS").lower()
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

# GTest sharding
SHARD_INDEX = os.getenv("SHARD_INDEX", 1)
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", 1)
environ_vars = os.environ.copy()
# For display purposes in the GitHub Action UI, the shard array is 1th indexed. However for shard indexes, we convert it to 0th index.
environ_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
environ_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)

logging.basicConfig(level=logging.INFO)

# If smoke tests are enabled, we run smoke tests only.
# Otherwise, we run the normal test suite
test_type = os.getenv("TEST_TYPE", "full")

# Only run regression test set for Windows or gfx11XX
if AMDGPU_FAMILIES.startswith("gfx11") or platform == "windows":
    test_type = "regression"

test_subdir = ""
timeout = "3600"
if test_type == "smoke":
    test_subdir = "/smoke"
    timeout = "300"
elif test_type == "regression":
    test_subdir = "/regression"
    timeout = "900"

cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/rocwmma{test_subdir}",
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
