import logging
import os
import shlex
import subprocess
from pathlib import Path
import multiprocessing

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
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

# DIAGNOSTIC: Log CPU allocation information to understand CI environment
logging.info("=== CPU Allocation Diagnostics ===")
logging.info(f"multiprocessing.cpu_count(): {multiprocessing.cpu_count()}")
logging.info(f"SLURM_CPUS_PER_TASK: {os.getenv('SLURM_CPUS_PER_TASK', 'NOT SET')}")
logging.info(f"SLURM_CPUS_ON_NODE: {os.getenv('SLURM_CPUS_ON_NODE', 'NOT SET')}")
logging.info(
    f"SLURM_JOB_CPUS_PER_NODE: {os.getenv('SLURM_JOB_CPUS_PER_NODE', 'NOT SET')}"
)
logging.info(f"OMP_NUM_THREADS (current): {os.getenv('OMP_NUM_THREADS', 'NOT SET')}")

# Try to detect cgroup CPU quota (container limits)
try:
    with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r") as f:
        quota = int(f.read().strip())
    with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r") as f:
        period = int(f.read().strip())
    if quota > 0:
        cgroup_cpus = quota // period
        logging.info(f"cgroup CPU quota: {cgroup_cpus} CPUs")
except:
    logging.info("cgroup CPU quota: Unable to detect")

logging.info("=== End Diagnostics ===")

# Set OMP_NUM_THREADS to prevent AOCL thread oversubscription
# CI environment reports 256 system cores but containers are limited to ~32-64 allocated cores.
# Standard detection methods (multiprocessing.cpu_count(), cgroup, SLURM vars) all fail
# to detect the actual allocation, so we use a conservative value.
#
# AOCL performance degrades 60-100x with thread oversubscription. Better to under-utilize
# than over-subscribe. Based on testing: 20 threads on 24-core allocation = 60x speedup.
#
# Setting to 48 threads assumes CI allocates 50-64 cores (leaving ~4-16 for system threads).
# This is conservative - if allocation is smaller, we under-utilize; if larger, we leave headroom.
if "OMP_NUM_THREADS" not in environ_vars:
    conservative_thread_count = 48
    environ_vars["OMP_NUM_THREADS"] = str(conservative_thread_count)
    logging.info(
        f"Setting OMP_NUM_THREADS={conservative_thread_count} (conservative for CI to prevent AOCL oversubscription)"
    )

# If smoke tests are enabled, we run smoke tests only.
# Otherwise, we run the normal test suite
test_type = os.getenv("TEST_TYPE", "full")
if test_type == "smoke":
    test_filter = ["--yaml", f"{THEROCK_BIN_DIR}/rocblas_smoke.yaml"]
else:
    # only running smoke tests due to openBLAS issue: https://github.com/ROCm/TheRock/issues/1605
    test_filter = ["--yaml", f"{THEROCK_BIN_DIR}/rocblas_smoke.yaml"]

cmd = [f"{THEROCK_BIN_DIR}/rocblas-test"] + test_filter
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    env=environ_vars,
    check=True,
)
