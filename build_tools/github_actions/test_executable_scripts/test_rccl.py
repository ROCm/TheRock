import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path
import pytest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
from github_actions_utils import get_visible_gpu_count

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
logging.basicConfig(level=logging.INFO)


class TestRCCL:
    def test_rccl_unittests(self):
        # Executing rccl gtest from rccl repo
        environ_vars = os.environ.copy()
        # Expect at least 2 GPUs for RCCL collectives
        gpu_count = get_visible_gpu_count(
            env=environ_vars, therock_bin_dir=THEROCK_BIN_DIR
        )
        logging.info(f"Visible GPU count: {gpu_count}")

        if gpu_count < 2:
            pytest.skip("Skipping RCCL unit tests: <2 GPUs visible")
        environ_vars["HIP_VISIBLE_DEVICES"] = "2,3"
        environ_vars["UT_MIN_GPUS"] = "2"
        environ_vars["UT_MAX_GPUS"] = "2"
        environ_vars["UT_POW2_GPUS"] = "1"
        environ_vars["UT_PROCESS_MASK"] = "1"
        cmd = [f"{THEROCK_BIN_DIR}/rccl-UnitTests"]
        logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
        result = subprocess.run(cmd, cwd=THEROCK_DIR, check=False, env=environ_vars)
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
