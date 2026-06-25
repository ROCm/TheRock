#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
===============================================================================
AMDSMI Test Runner (Manual Execution Only)

This script is NOT part of automated CI runs.

`amdsmitst` requires GPU device access (/dev/kfd, /dev/dri), elevated
permissions, and execution on a ROCm-enabled system. GitHub-hosted CI
environments do not expose these capabilities, so this script must be run
manually by developers inside a privileged ROCm environment or container.

Usage:
    python test_amdsmi.py

===============================================================================
"""

import logging
import os
import platform
import shlex
import subprocess
from pathlib import Path


def is_windows():
    return "windows" == platform.system().lower()


logging.basicConfig(level=logging.INFO)

SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

AMDSMITST_BIN = (
    THEROCK_DIR / "build" / "share" / "amd_smi" / "tests" / "amdsmitst"
).resolve()

platform_key = "windows" if is_windows() else "linux"
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")

# -----------------------------
# GTest sharding
# -----------------------------
SHARD_INDEX = os.getenv("SHARD_INDEX", "1")
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", "1")

environ_vars = os.environ.copy()
environ_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
environ_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)
test_type = os.getenv("TEST_TYPE", "standard")

if test_type == "quick":
    logging.info("Running quick tests only for amdsmitst")
    test_filter = ["--gtest_filter=AmdSmiDynamicMetricTest.*"]

else:
    logging.info("Running standard amdsmitst test suite (include + exclude filter)")

    include_tests = [
        "amdsmitstReadOnly.*",
        "amdsmitstReadWrite.FanReadWrite",
        "amdsmitstReadWrite.TestOverdriveReadWrite",
        "amdsmitstReadWrite.TestPciReadWrite",
        "amdsmitstReadWrite.TestPowerReadWrite",
        "amdsmitstReadWrite.TestPerfCntrReadWrite",
        "amdsmitstReadWrite.TestEvtNotifReadWrite",
        "AmdSmiDynamicMetricTest.*",
    ]

    exclude_tests = [
        "amdsmitstReadOnly.TempRead",
        "amdsmitstReadOnly.TestFrequenciesRead",
        "amdsmitstReadWrite.TestPowerReadWrite",
    ]

    # -----------------------------
    # Arch-specific ignores (CI parity)
    # -----------------------------
    TESTS_TO_IGNORE = {
        "gfx90a": {
            "linux": [
                "amdsmitstReadOnly.TestSysInfoRead",
                "amdsmitstReadOnly.TestIdInfoRead",
                "amdsmitstReadWrite.TestPciReadWrite",
            ]
        },
        "gfx110X-all": {
            "linux": [
                # gfx110X runs on a non-privileged runner (no --privileged
                # container), so exclude write/set tests that require elevated
                # device access. Leaves the read-only + dynamic-metric subset.
                "amdsmitstReadWrite.FanReadWrite",
                "amdsmitstReadWrite.TestOverdriveReadWrite",
                "amdsmitstReadWrite.TestPciReadWrite",
                "amdsmitstReadWrite.TestPerfCntrReadWrite",
                "amdsmitstReadWrite.TestEvtNotifReadWrite",
            ]
        },
        "gfx103X-all": {
            "linux": [
                "amdsmitstReadWrite.FanReadWrite",
            ]
        },
    }

    if (
        AMDGPU_FAMILIES in TESTS_TO_IGNORE
        and platform_key in TESTS_TO_IGNORE[AMDGPU_FAMILIES]
    ):
        ignored_tests = TESTS_TO_IGNORE[AMDGPU_FAMILIES][platform_key]
        logging.info(f"Adding arch-specific excludes: {ignored_tests}")
        exclude_tests.extend(ignored_tests)

    logging.info(f"AMDGPU_FAMILIES={AMDGPU_FAMILIES}")
    logging.info(f"platform_key={platform_key}")
    logging.info(f"Final exclude_tests={exclude_tests}")

    # -----------------------------
    # Build final filter
    # -----------------------------
    gtest_filter = f"{':'.join(include_tests)}:-{':'.join(exclude_tests)}"
    test_filter = [f"--gtest_filter={gtest_filter}"]

# -----------------------------
# Build command
# -----------------------------
cmd = [str(AMDSMITST_BIN)] + test_filter

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

if not AMDSMITST_BIN.exists():
    raise FileNotFoundError(f"amdsmitst not found at {AMDSMITST_BIN}")

if not os.access(AMDSMITST_BIN, os.X_OK):
    raise PermissionError(f"amdsmitst is not executable: {AMDSMITST_BIN}")

# -----------------------------
# Run tests
# -----------------------------
subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    env=environ_vars,
    check=True,
)
