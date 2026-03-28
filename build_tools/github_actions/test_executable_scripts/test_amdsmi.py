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

import pytest

pytestmark = pytest.mark.skip("Manual execution only — requires GPU device access")
import logging
import os
import shlex
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)

SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

TESTS_DIR = (THEROCK_DIR / "build" / "share" / "amd_smi" / "tests").resolve()
AMDSMITST_BIN = TESTS_DIR / "amdsmitst"


def get_asic_exclude_filter(test_dir):
    """Source amdsmitst.exclude and detect_asic_filter.sh, return GTEST_EXCLUDE."""
    exclude_script = test_dir / "amdsmitst.exclude"
    detect_script = test_dir / "detect_asic_filter.sh"

    if not exclude_script.exists():
        logging.warning(f"amdsmitst.exclude not found in {test_dir}")
        return ""
    if not detect_script.exists():
        logging.warning(f"detect_asic_filter.sh not found in {test_dir}")
        return ""

    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{exclude_script}" && source "{detect_script}" && echo "$GTEST_EXCLUDE"',
        ],
        capture_output=True,
        text=True,
        cwd=str(test_dir),
    )

    if result.returncode != 0:
        logging.warning(
            f"ASIC detection failed (rc={result.returncode}): {result.stderr.strip()}"
        )
        return ""

    gtest_exclude = result.stdout.strip()
    if gtest_exclude:
        logging.info(f"ASIC exclude filter: {gtest_exclude}")
    else:
        logging.info("ASIC detection returned no exclusions")
    return gtest_exclude


# -----------------------------
# GTest sharding
# -----------------------------
SHARD_INDEX = os.getenv("SHARD_INDEX", "1")
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", "1")
env = os.environ.copy()

# Convert to 0-based index for GTest
env["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
env["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)

# -----------------------------
# Test filtering
# -----------------------------
# If quick mode is enabled, run minimal suite (only dynamic metric tests)
test_type = os.getenv("TEST_TYPE", "full")

if test_type == "quick":
    include_tests = [
        "AmdSmiDynamicMetricTest.*",
    ]
    include_filter = ":".join(include_tests)
    gtest_filter_arg = [f"--gtest_filter={include_filter}"]
    logging.info(f"Quick mode: include filter = {include_filter}")
else:
    # Full test mode: negative-only filter matching upstream CI pattern
    # (./amdsmitst --gtest_filter="-${GTEST_EXCLUDE}")

    # Manual exclusions — always applied regardless of ASIC
    exclude_tests = [
        "amdsmitstReadOnly.TempRead",
        "amdsmitstReadOnly.TestFrequenciesRead",
        "amdsmitstReadWrite.TestPowerReadWrite",
    ]

    # Merge ASIC-specific exclusions from detect_asic_filter.sh
    asic_exclude = get_asic_exclude_filter(TESTS_DIR)
    if asic_exclude:
        asic_tests = [t for t in asic_exclude.split(":") if t]
        for test in asic_tests:
            if test not in exclude_tests:
                exclude_tests.append(test)
        logging.info(
            f"Combined exclude list ({len(exclude_tests)} entries): {exclude_tests}"
        )

    exclude_filter = f"-{':'.join(exclude_tests)}"
    gtest_filter_arg = [f"--gtest_filter={exclude_filter}"]
    logging.info(f"Full mode: exclude filter = {exclude_filter}")

# -----------------------------
# Build command
# -----------------------------
cmd = [str(AMDSMITST_BIN)] + gtest_filter_arg

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

# -----------------------------
# Run tests
# -----------------------------
subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    env=env,
    check=True,
)
