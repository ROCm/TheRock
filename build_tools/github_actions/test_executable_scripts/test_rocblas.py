import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

# Importing is_asan from github_actions_utils.py
sys.path.append(str(THEROCK_DIR / "build_tools" / "github_actions"))
from github_actions_utils import is_asan

# GTest sharding
SHARD_INDEX = os.getenv("SHARD_INDEX", 1)
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", 1)
environ_vars = os.environ.copy()
# For display purposes in the GitHub Action UI, the shard array is 1th indexed. However for shard indexes, we convert it to 0th index.
environ_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
environ_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)

if is_asan():
    environ_vars["HSA_XNACK"] = "1"

# Limit OpenBLAS/OpenMP threads in CI to avoid overallocation (e.g. 150 threads on
# high core-count visibility) which can cause contention and 2x+ slower test runs.
environ_vars["OPENBLAS_NUM_THREADS"] = "48"
environ_vars["OMP_NUM_THREADS"] = "48"

logging.basicConfig(level=logging.INFO)

# Common args for CI (xml output for reporting, color for logs)
cmd = [
    f"{THEROCK_BIN_DIR}/rocblas-test",
    "--gtest_output=xml",
    "--gtest_color=yes",
]

# If smoke tests are enabled, use the YAML filter. Otherwise run quick + pre_checkin (exclude known_bug),
# matching rocBLAS upstream CI: *quick*:*pre_checkin*-*known_bug*
test_type = os.getenv("TEST_TYPE", "full")
if test_type == "smoke":
    cmd += ["--yaml", f"{THEROCK_BIN_DIR}/rocblas_smoke.yaml"]
else:
    cmd += ["--gtest_filter=*quick*:*pre_checkin*-*known_bug*"]

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    env=environ_vars,
    check=True,
)
