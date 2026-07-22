#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
===============================================================================
AMDSMI Test Runner

Runs the `amdsmitst` GTest binary shipped with the amd-smi component.

ASIC-specific exclusions are delegated to the amd-smi test tree itself
(amdsmitst.exclude + detect_asic_filter.sh), so this runner stays agnostic to
which tests each ASIC skips. Environment tiering (privileged docker /
unprivileged docker / baremetal) is detected here and surfaced to the binary
via AMDSMI_NON_PRIVILEGED so state-modifying tests skip themselves when the
runtime lacks the privileges they need.

Usage:
    python test_amdsmi.py

===============================================================================
"""

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

    # check=True: a detection failure is a hard error (raises
    # CalledProcessError and ends the run) rather than silently degrading to an
    # unfiltered test set. capture_output is kept because we still need the
    # GTEST_EXCLUDE value echoed on stdout.
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{exclude_script}" && source "{detect_script}" && echo "$GTEST_EXCLUDE"',
        ],
        capture_output=True,
        text=True,
        cwd=str(test_dir),
        check=True,
    )

    gtest_exclude = result.stdout.strip()
    if gtest_exclude:
        logging.info(f"ASIC exclude filter: {gtest_exclude}")
    else:
        logging.info("ASIC detection returned no exclusions")
    return gtest_exclude


def _cgroup_mentions_container():
    try:
        cgroup = Path("/proc/1/cgroup").read_text()
    except OSError:
        return False
    return any(marker in cgroup for marker in ("docker", "kubepods", "containerd"))


def _in_container():
    # /.dockerenv (docker) and /run/.containerenv (podman) are the standard
    # marker files; the `container` env var is set by systemd-nspawn and several
    # OCI runtimes. Fall back to cgroup inspection for runtimes that set none.
    if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
        return True
    if os.environ.get("container"):
        return True
    return _cgroup_mentions_container()


def detect_privilege_tier():
    """Classify the runtime as 'baremetal', 'privileged', or 'unprivileged'.

    amdsmitst gates state-modifying tests on is_sudo_user() (uid == 0). Inside an
    unprivileged docker container the process is uid 0 yet lacks the full
    capability set a privileged container or baremetal root has, so is_sudo_user()
    alone cannot tell the two apart. Inspect the effective capability mask to
    distinguish them so the caller can set AMDSMI_NON_PRIVILEGED accordingly.
    """
    if not _in_container():
        return "baremetal"

    # CapEff is a 64-bit hex mask in /proc/self/status. A privileged container
    # (or --cap-add=ALL) exposes the full mask; unprivileged containers get the
    # trimmed default docker set. CAP_SYS_ADMIN (bit 21) is the practical marker
    # for "can touch the sysfs/debugfs knobs the write tests need".
    cap_sys_admin = 21
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("CapEff:"):
                cap_eff = int(line.split()[1], 16)
                has_admin = cap_eff & (1 << cap_sys_admin)
                return "privileged" if has_admin else "unprivileged"
    except (OSError, ValueError):
        pass
    # Unknown capability state inside a container: treat as unprivileged so we
    # err on the side of skipping state-modifying tests rather than hanging.
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
# Environment tiering
# -----------------------------
# Surface the privilege tier to the binary. Never override an explicit setting
# from the caller; only add AMDSMI_NON_PRIVILEGED when we detect an unprivileged
# runtime that cannot safely run state-modifying tests.
privilege_tier = detect_privilege_tier()
logging.info(f"Detected privilege tier: {privilege_tier}")
if "AMDSMI_NON_PRIVILEGED" in environ_vars:
    logging.info(
        "AMDSMI_NON_PRIVILEGED already set by caller "
        f"({environ_vars['AMDSMI_NON_PRIVILEGED']!r}); leaving as-is"
    )
elif privilege_tier == "unprivileged":
    environ_vars["AMDSMI_NON_PRIVILEGED"] = "1"
    logging.info("Unprivileged runtime: setting AMDSMI_NON_PRIVILEGED=1")

# -----------------------------
# Test filtering
# -----------------------------
test_type = os.getenv("TEST_TYPE", "standard")

if test_type == "quick":
    # Rename-agnostic: matches both the legacy AmdSmiDynamicMetricTest suite and
    # the GpuUnit suite it is being renamed to. gtest treats unmatched patterns
    # as a no-op, so listing both is safe before and after the rename.
    include_tests = [
        "AmdSmiDynamicMetricTest.*",
        "*Unit*",
    ]
    include_filter = ":".join(include_tests)
    gtest_filter_arg = [f"--gtest_filter={include_filter}"]
    logging.info(f"Quick mode: include filter = {include_filter}")
else:
    # Full mode: rename-agnostic positive whitelist minus the sourced exclusions.
    # Listing both the legacy amdsmitst* suites and the new *Functional*/*Unit*
    # suites keeps the logical test set stable across the suite rename without
    # automatically pulling in unrelated new suites (NIC/IFoE/etc.).
    include_tests = [
        "amdsmitstReadOnly.*",
        "amdsmitstReadWrite.*",
        "AmdSmiDynamicMetricTest.*",
        "*FunctionalReadOnly.*",
        "*FunctionalReadWrite.*",
        "*Unit*",
    ]

    # Manual exclusions — always applied regardless of ASIC. Listed under both
    # the legacy and renamed suite names so they survive the rename.
    exclude_tests = [
        "amdsmitstReadOnly.TempRead",
        "amdsmitstReadOnly.TestFrequenciesRead",
        "amdsmitstReadWrite.TestPowerReadWrite",
        "GpuFunctionalReadOnly.TempRead",
        "GpuFunctionalReadOnly.TestFrequenciesRead",
        "GpuFunctionalReadWrite.TestPowerReadWrite",
    ]

    # ASIC- and environment-specific exclusions come from detect_asic_filter.sh,
    # which sources the exclude table shipped alongside the binary, so the names
    # always match the binary being tested.
    asic_exclude = get_asic_exclude_filter(TESTS_DIR)
    if asic_exclude:
        for test in asic_exclude.split(":"):
            if test and test not in exclude_tests:
                exclude_tests.append(test)
        logging.info(
            f"Combined exclude list ({len(exclude_tests)} entries): {exclude_tests}"
        )

    # EXPERIMENT (users/kbillaka): un-exclude the write/set tests that CI normally
    # filters out on gfx1151, to observe their real pass/skip behaviour on the
    # non-privileged strix runner. Privilege detection above is left intact, so
    # these tests still self-skip their state-modifying paths when the runtime
    # lacks the capabilities — we only drop the hard gtest-filter exclusion.
    # Matches by test-case name so it covers both the legacy amdsmitst* and the
    # renamed GpuFunctional* suites, and both the global and ASIC exclude sources.
    UNEXCLUDE_TESTCASES = {"TestPowerReadWrite", "TestFrequenciesReadWrite"}
    _before = len(exclude_tests)
    exclude_tests = [
        t for t in exclude_tests if t.rsplit(".", 1)[-1] not in UNEXCLUDE_TESTCASES
    ]
    logging.info(
        f"Experiment: un-excluded {_before - len(exclude_tests)} write test(s) "
        f"({sorted(UNEXCLUDE_TESTCASES)}); {len(exclude_tests)} excludes remain"
    )

    gtest_filter = f"{':'.join(include_tests)}:-{':'.join(exclude_tests)}"
    gtest_filter_arg = [f"--gtest_filter={gtest_filter}"]
    logging.info(f"Full mode: filter = {gtest_filter}")

# -----------------------------
# Build command
# -----------------------------
cmd = [str(AMDSMITST_BIN)] + gtest_filter_arg

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
