# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check that Triton device printing completes against the installed HIP runtime."""

import importlib.util
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    platform.system() != "Linux", reason="Requires the ROCm Triton backend"
)
def test_triton_hostcall(tmp_path):
    if importlib.util.find_spec("triton") is None:
        pytest.skip("Triton was not built")
    env = dict(
        os.environ,
        TRITON_CACHE_DIR=str(tmp_path / "triton-cache"),
        TRITON_INTERPRET="0",
    )
    # A protocol mismatch can stall synchronization. Isolate it from the rest of
    # the smoke suite and bound the wait, including on machines without pytest-timeout.
    kernel_script = Path(__file__).resolve().parent.parent / "triton_hostcall.py"
    result = subprocess.run(
        [sys.executable, str(kernel_script)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )
    output = result.stdout.decode(errors="replace")
    assert result.returncode == 0, output
    assert "TheRock hostcall smoke" in output, output
    assert "hostcall synchronized" in output, output
