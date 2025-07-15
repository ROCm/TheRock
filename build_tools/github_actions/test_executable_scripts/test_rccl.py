import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO)

# Executing rccl gtest from rccl repo
cmd = [f"{THEROCK_BIN_DIR}/rccl-UnitTests"]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=True,
)

# Executing rccl performance and correctness tests from rccl-tests repo
executables = [
    "all_gather_perf",
    "alltoallv_perf",
    "broadcast_perf",
    "alltoall_perf",
    "all_reduce_perf",
    "reduce_perf",
    "hypercube_perf",
    "gather_perf",
    "scatter_perf",
    "sendrecv_perf",
    "reduce_scatter_perf",
]

for executable in executables:
    cmd = [f"{THEROCK_BIN_DIR}/{executable}"]
    logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
    subprocess.run(
        cmd,
        cwd=THEROCK_DIR,
        check=True,
    )
