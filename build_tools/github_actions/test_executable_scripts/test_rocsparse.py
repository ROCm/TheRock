# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = Path(os.getenv("THEROCK_BIN_DIR")).resolve()
OUTPUT_ARTIFACTS_DIR = Path(os.getenv("OUTPUT_ARTIFACTS_DIR")).resolve()
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent.resolve()

import sys
sys.path.append(str(THEROCK_DIR / "build_tools" / "github_actions"))
from github_actions_api import is_host_asan

if is_host_asan():
    print(
        "SKIP: host-ASAN build (THEROCK_SANITIZER=HOST_ASAN) does not produce "
        "GPU math library test binaries. Run with linux-release-asan for device tests."
    )
    sys.exit(0)

logging.basicConfig(level=logging.INFO)

# GTest sharding
SHARD_INDEX = os.getenv("SHARD_INDEX", 1)
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", 1)
environ_vars = os.environ.copy()
# For display purposes in the GitHub Action UI, the shard array is 1th indexed. However for shard indexes, we convert it to 0th index.
environ_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
environ_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)

# If quick tests are enabled, we run quick tests only.
# Otherwise, we run the normal test suite
test_type = os.getenv("TEST_TYPE", "full")
if test_type == "quick":
    test_filter = [
        "--yaml",
        f"{THEROCK_DIR}/build/share/rocsparse/test/rocsparse_smoke.yaml",
    ]
else:
    # TODO(#2616): Enable full tests once known test issues are resolved
    test_filter = [
        "--yaml",
        f"{THEROCK_DIR}/build/share/rocsparse/test/rocsparse_smoke.yaml",
    ]

cmd = [
    f"{THEROCK_BIN_DIR}/rocsparse-test",
    "--matrices-dir",
    f"{OUTPUT_ARTIFACTS_DIR}/clients/matrices/",
] + test_filter
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)
