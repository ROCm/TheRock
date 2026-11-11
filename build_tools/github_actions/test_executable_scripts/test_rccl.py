# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import re
import shlex
import subprocess
from pathlib import Path
import pytest

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
logging.basicConfig(level=logging.INFO)


def _get_visible_gpu_count(env=None, therock_bin_dir: str | None = None) -> int:
    rocminfo = Path(therock_bin_dir) / "rocminfo"
    rocminfo_cmd = str(rocminfo) if rocminfo.exists() else "rocminfo"

    result = subprocess.run(
        [rocminfo_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        check=False,
    )

    pattern = re.compile(r"^\s*Name:\s+gfx[0-9a-z]+(?:-strict)?$", re.IGNORECASE)

    gpu_count = sum(
        1 for line in result.stdout.splitlines() if pattern.match(line.strip())
    )
    # rocminfo enumerates HSA agents and may not honor HIP's narrower mask.
    # Keep the caller's isolation and count the GPUs its HIP process can use.
    gpu_env = os.environ if env is None else env
    visible_devices = gpu_env.get("HIP_VISIBLE_DEVICES")
    if visible_devices is None:
        visible_devices = gpu_env.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        devices = [device.strip() for device in visible_devices.split(",")]
        mask_count = 0
        for device in devices:
            if not device or device.startswith("-"):
                break
            mask_count += 1
        gpu_count = min(gpu_count, mask_count)
    return gpu_count


def _multi_gpu_coverage_requested() -> bool:
    return os.getenv("MULTI_GPU_TEST") == "1" and os.getenv(
        "TEST_TYPE", "standard"
    ) in ("comprehensive", "full")


def _get_test_gpu_count(environ_vars: dict[str, str]) -> int:
    gpu_count = _get_visible_gpu_count(
        env=environ_vars, therock_bin_dir=THEROCK_BIN_DIR
    )
    logging.info(f"Visible GPU count: {gpu_count}")
    if _multi_gpu_coverage_requested() and gpu_count < 2:
        pytest.fail(
            "RCCL comprehensive/full multi-GPU tests require at least 2 visible "
            f"GPUs; found {gpu_count}"
        )
    return gpu_count


class TestRCCL:
    def test_rccl_unittests(self):
        # Executing rccl gtest from rccl repo
        environ_vars = os.environ.copy()
        # Expect at least 2 GPUs for RCCL collectives
        gpu_count = _get_test_gpu_count(environ_vars)

        if gpu_count < 2:
            pytest.skip("Skipping RCCL unit tests: <2 GPUs visible")
        environ_vars["UT_MIN_GPUS"] = "2"
        environ_vars["UT_MAX_GPUS"] = (
            str(gpu_count) if _multi_gpu_coverage_requested() else "2"
        )
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
        if _multi_gpu_coverage_requested():
            gpu_count = _get_test_gpu_count(os.environ.copy())
            cmd.extend(["-g", str(gpu_count)])
        logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=THEROCK_DIR,
            check=False,
        )
        assert result.returncode == 0
