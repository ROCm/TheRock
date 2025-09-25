import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

PLATFORM = os.getenv("PLATFORM")
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")

logging.basicConfig(level=logging.INFO)

# GTest sharding
SHARD_INDEX = os.getenv("SHARD_INDEX", 1)
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", 1)
envion_vars = os.environ.copy()
# For display purposes in the GitHub Action UI, the shard array is 1th indexed. However for shard indexes, we convert it to 0th index.
envion_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
envion_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)

exclude_tests_filter = ""

tests_to_exclude = {
    # Related issue: https://ontrack-internal.amd.com/browse/SWDEV-557164
    "gfx1151": {
        "windows": [
            "quick/bsrsv.level2/*",
            "quick/coomv.level2/*",
            "quick/csrmv.level2/*",
        ]
    }
}

if (
    AMDGPU_FAMILIES in tests_to_exclude
    and PLATFORM in tests_to_exclude[AMDGPU_FAMILIES]
):
    exclude_tests_filter += "-" + ":".join(tests_to_exclude[AMDGPU_FAMILIES][PLATFORM])

cmd = [
    f"{THEROCK_BIN_DIR}/rocsparse-test",
    "--gtest_filter=*quick*" + exclude_tests_filter,
    "--matrices-dir",
    f"{OUTPUT_ARTIFACTS_DIR}/clients/matrices/",
]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=envion_vars)
