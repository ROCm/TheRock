import logging
import os
import shlex
import shutil
import subprocess
from pathlib import Path
import pytest

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
logging.basicConfig(level=logging.INFO)

amd_smi_exists = shutil.which("amd-smi") is not None

pytestmark = pytest.mark.skipif(
    not amd_smi_exists, reason="amd-smi not available, skipping RCCL tests"
)


class TestRCCL:
    def test_rccl_unittests(self):
        # Executing rccl gtest from rccl repo
        cmd = [f"{THEROCK_BIN_DIR}/rccl-UnitTests"]
        logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=THEROCK_DIR,
            check=False,
        )
        assert result.returncode == 0

    # Executing rccl performance and correctness tests from rccl-tests repo
    @pytest.mark.parametrize(
        "executable",
        [
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
        ],
    )
    def test_rccl_correctness_tests(self, executable):
        cmd = [f"{THEROCK_BIN_DIR}/{executable}"]
        logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=THEROCK_DIR,
            check=False,
        )
        assert result.returncode == 0
