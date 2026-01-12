import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))
from github_actions_utils import get_first_gpu_architecture

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")

logging.basicConfig(level=logging.INFO)

# GTest sharding
SHARD_INDEX = os.getenv("SHARD_INDEX", 1)
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", 1)
environ_vars = os.environ.copy()
# For display purposes in the GitHub Action UI, the shard array is 1th indexed.
# For shard indexes, we convert to 0th index.
environ_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
environ_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)

gpu_arch = get_first_gpu_architecture(env=environ_vars, therock_bin_dir=THEROCK_BIN_DIR)
logging.info(f"Detected GPU architecture: {gpu_arch}")

cwd_dir = Path(THEROCK_BIN_DIR) / gpu_arch
cmd = ["./rocrtst64"]

# If smoke tests are enabled, run smoke tests only. Otherwise, run the full suite.
SMOKE_TESTS = [
    "rocrtst.Test_Example",
    "rocrtstFunc.MemoryAccessTests",
    "rocrtstFunc.GroupMemoryAllocationTest",
    "rocrtstFunc.MemoryAllocateAndFreeTest",
    "rocrtstFunc.Memory_Alignment_Test",
    "rocrtstFunc.Concurrent_Init_Test",
    "rocrtstFunc.Concurrent_Init_Shutdown_Test",
    "rocrtstFunc.Reference_Count",
    "rocrtstFunc.Signal_Create_Concurrently",
    "rocrtstFunc.Signal_Destroy_Concurrently",
    "rocrtstFunc.IPC",
    "rocrtstFunc.AgentProp_UUID",
    "rocrtstFunc.Deallocation_Notifier_Test",
    "rocrtstFunc.Memory_Atomic_Add_Test",
    "rocrtstFunc.Memory_Atomic_Xchg_Test",
]
test_type = os.getenv("TEST_TYPE", "full")
if test_type == "smoke":
    environ_vars["GTEST_FILTER"] = ":".join(SMOKE_TESTS)

logging.info(f"++ Exec [{cwd_dir}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=cwd_dir, check=True, env=environ_vars)
