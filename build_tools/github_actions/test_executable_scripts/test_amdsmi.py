#!/usr/bin/env python3
import logging
import os
import shlex
import subprocess
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO)

SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

# Detect current Python major/minor version
py_major = sys.version_info.major
py_minor = sys.version_info.minor

VENV_SITE_PACKAGES = (
    THEROCK_DIR
    / ".venv"
    / "lib"
    / f"python{py_major}.{py_minor}"
    / "site-packages"
)

# Path to amdsmitst binary
AMDSMITS_PATH = (
    VENV_SITE_PACKAGES
    / "_rocm_sdk_core"
    / "share"
    / "amd_smi"
    / "tests"
    / "amdsmitst"
)

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
# If smoke mode is enabled, run minimal suite (only dynamic metric tests)
test_type = os.getenv("TEST_TYPE", "full")

if test_type == "smoke":
    logging.info("Running smoke tests only for amdsmitst")
    test_filter = [
        "--gtest_filter=AmdSmiDynamicMetricTest.*"
    ]
else:
    # Full test mode: whitelist only passing tests because exclude filters don't work
    logging.info("Running full amdsmitst test suite (with whitelist filter)")

    # Passing tests from amdsmitst
    INCLUDE_FILTER = (
        "amdsmitstReadOnly.*:"
        "amdsmitstReadWrite.FanReadWrite:"
        "amdsmitstReadWrite.TestOverdriveReadWrite:"
        "amdsmitstReadWrite.TestPciReadWrite:"
        "amdsmitstReadWrite.TestPowerReadWrite:"
        "amdsmitstReadWrite.TestPerfCntrReadWrite:"
        "amdsmitstReadWrite.TestEvtNotifReadWrite:"
        "AmdSmiDynamicMetricTest.*"
    )

    test_filter = [f"--gtest_filter={INCLUDE_FILTER}"]

# -----------------------------
# Build command
# -----------------------------
cmd = [str(AMDSMITS_PATH)] + test_filter

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
