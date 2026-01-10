import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO)

# GTest sharding
SHARD_INDEX = os.getenv("SHARD_INDEX", 1)
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", 1)
envion_vars = os.environ.copy()
# For display purposes in the GitHub Action UI, the shard array is 1th indexed. However for shard indexes, we convert it to 0th index.
envion_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
envion_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)

cmd = [
    f"{THEROCK_BIN_DIR}/rocsolver-test",
]

# If smoke tests are enabled, we run smoke tests only.
# Otherwise, we run the normal test suite
# Test filter patterns retrieved from https://github.com/ROCm/rocm-libraries/blob/a18b17eef6c24bcd4bcf8dd6a0e36325cbcd11a7/projects/rocsolver/rtest.xml
test_type = os.getenv("TEST_TYPE", "full")
if test_type == "smoke":
    smoke_tests = [
        "checkin*BDSQR*",
        "checkin*STEBZ*",
        "checkin*STEIN*",
        "checkin*STERF*",
        "checkin*STEQR*",
        "checkin*SYEVJ*",
        "checkin*HEEVJ*",
        "checkin*LARFG*",
        "checkin*LARF*",
        "checkin*LARFT*",
        "checkin*GETF2*",
        "checkin*POTF2*",
        "checkin*GEQR2*",
        "checkin*GELQ2*",
        "checkin*SPLITLU*",
        "checkin*REFACTLU*",
        "checkin*REFACTCHOL*",
    ]
    cmd.extend([f"--gtest_filter={':'.join(smoke_tests)}-*LARFB*:*known_bug*"])
else:
    cmd.extend(
        ["--gtest_filter=checkin*-*known_bug*:checkin_lapack/SYGVDX_INPLACE.__float/41"]
    )

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=envion_vars)
