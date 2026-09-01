#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
===============================================================================
AMDCUID Test Runner

Runs the `amdcuid_test` GTest binary shipped with the core-amdcuid component.

Tests are split by privilege level:
  - cuidtstUnprivileged.*  — safe to run without elevated privileges
  - cuidtstPrivileged.*    — requires root or CAP_SYS_ADMIN (skipped otherwise)

Usage:
    python test_amdcuid.py

===============================================================================
"""

import logging
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO)
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

TESTS_DIR = (THEROCK_DIR / "build" / "share" / "amdcuid" / "tests").resolve()
AMDCUID_TST_BIN = TESTS_DIR / "amdcuid_test"


def _cgroup_mentions_container():
    try:
        cgroup = Path("/proc/1/cgroup").read_text()
    except OSError:
        return False
    return any(marker in cgroup for marker in ("docker", "kubepods", "containerd"))


def _in_container():
    if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
        return True
    if os.environ.get("container"):
        return True
    return _cgroup_mentions_container()


def detect_privilege_tier():
    """Classify the runtime as 'baremetal', 'privileged', or 'unprivileged'."""
    if not _in_container():
        return "baremetal"

    cap_sys_admin = 21
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("CapEff:"):
                cap_eff = int(line.split()[1], 16)
                has_admin = cap_eff & (1 << cap_sys_admin)
                return "privileged" if has_admin else "unprivileged"
    except (OSError, ValueError):
        pass
    return "unprivileged"


# -----------------------------
# GTest sharding
# -----------------------------
SHARD_INDEX = os.getenv("SHARD_INDEX", "1")
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", "1")

environ_vars = os.environ.copy()
environ_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
environ_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)

# -----------------------------
# Hmac key provisioning
# -----------------------------
# Generate a temporary HMAC key so the library can compute derived CUIDs
# without needing the root-provisioned /etc/amdcuid/hmac_key.bin.
_tmp_key_dir = tempfile.mkdtemp(prefix="amdcuid_ci_")
_tmp_key_path = os.path.join(_tmp_key_dir, "hmac_key.bin")
with open(_tmp_key_path, "wb") as f:
    f.write(os.urandom(32))
environ_vars["AMDCUID_HMAC_KEY_PATH"] = _tmp_key_path

# -----------------------------
# Privilege detection
# -----------------------------
privilege_tier = detect_privilege_tier()
logging.info(f"Detected privilege tier: {privilege_tier}")

# Unprivileged containers cannot run cuidtstPrivileged tests (they touch
# /etc/amdcuid and sysfs paths that require CAP_SYS_ADMIN).
run_privileged = privilege_tier in ("baremetal", "privileged")

# -----------------------------
# Test filtering
# -----------------------------
if run_privileged:
    gtest_filter = "cuidtstUnprivileged.*:cuidtstPrivileged.*"
    logging.info("Privileged runtime: running unprivileged + privileged test suites")
else:
    gtest_filter = "cuidtstUnprivileged.*"
    logging.info("Unprivileged runtime: running unprivileged test suite only")

# -----------------------------
# Build command
# -----------------------------
cmd = [str(AMDCUID_TST_BIN), f"--gtest_filter={gtest_filter}"]

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

if not AMDCUID_TST_BIN.exists():
    raise FileNotFoundError(f"amdcuid_test not found at {AMDCUID_TST_BIN}")

if not os.access(AMDCUID_TST_BIN, os.X_OK):
    raise PermissionError(f"amdcuid_test is not executable: {AMDCUID_TST_BIN}")

# -----------------------------
# Run tests
# -----------------------------
subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    env=environ_vars,
    check=True,
)
