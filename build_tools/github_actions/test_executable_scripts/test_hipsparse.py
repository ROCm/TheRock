import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
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

environ_vars[
    "HIPSPARSE_CLIENTS_MATRICES_DIR"
] = f"{OUTPUT_ARTIFACTS_DIR}/clients/matrices/"

cmd = [f"{THEROCK_BIN_DIR}/hipsparse-test"]

test_type = os.getenv("TEST_TYPE", "full")
if test_type == "smoke":
    cmd.append("--gtest_filter=*spmv*:*spsv*:*spsm*:*spmm*:*csric0*:*csrilu0*")

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=True,
    env=environ_vars,
)
