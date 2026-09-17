#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run RCCL's exact CPU-only host-TSAN inventory without GPU devices."""

import hashlib
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from host_tsan_instrumentation import (
    native_host_tsan_environment,
    parse_gtest_listed_tests,
    require_direct_clang_tsan,
    require_gtest_execution,
    require_no_gpu_nodes,
)


DEATH_TEST_EXCLUSIONS = (
    "InitMicrotest.P2pSchedule_ZeroGroupSizeParam_DiesOnDivideByZero",
    "InitMicrotest.CommShrink_NullNewcomm_DiesOnNullDeref",
    "InitMicrotest.SetCommAbortFlags_NullChildDevUnderNonNullChildFlag_DiesOnNullDeref",
)

GTEST_INVENTORIES = (
    {
        "name": "rccl-UnitTestsMicro",
        "count": 110,
        "sha256": "0e6225d55fa67a0fa2a642b181a056e125b929be192ed37be3e08756fedccdaf",
        "executed": 110,
    },
    {
        "name": "rccl-UnitTestsMicroEnqueue",
        "count": 177,
        "sha256": "1ee3e97ee5e0d455ddcf072d094c7f4c633b9cdf2be4f43aadfbb3d7570a9778",
        "executed": 177,
    },
    {
        "name": "rccl-UnitTestsMicroInit",
        "count": 688,
        "sha256": "c08ad78a18c4211548fa8904f6d601d51768a2ab470d144926cd139d0fe7099e",
        "executed": 685,
        "exclude": DEATH_TEST_EXCLUSIONS,
    },
    {
        "name": "rccl-UnitTestsMicroInit-faultinj",
        "count": 688,
        "sha256": "c08ad78a18c4211548fa8904f6d601d51768a2ab470d144926cd139d0fe7099e",
        "executed": 685,
        "exclude": DEATH_TEST_EXCLUSIONS,
    },
    {
        "name": "rccl-UnitTestsMicroInit-uncached",
        "count": 687,
        "sha256": "1d4dcc58d78bfa3be9c0bb0345279d88d859e344c8ebf489448943080b671dbb",
        "executed": 684,
        "exclude": DEATH_TEST_EXCLUSIONS,
    },
)

NET_TELEMETRY_BINARY = "rccl-UnitTestsNetTelemetry"
NET_TELEMETRY_CASES = (
    "testBlockIndex",
    "testGrowth",
    "testOverOldCeiling",
    "testChannelSanityBound",
    "testInvariantSingle",
    "testAggregateAcrossBlocks",
    "testCacheLineAlignment",
    "testHandleResolution",
    "testHandleEquivalence",
    "testHandleStableAcrossGrowth",
    "testLatencyBucket",
    "testConfigLayout",
    "testSamplingSelection",
    "testSamplingSentinel",
    "testSamplingCounters",
    "testSamplingDefaultIsIdentity",
    "testConcurrent",
)

_PASSED_RE = re.compile(r"\[\s*PASSED\s*\]\s+(\d+) tests?\.")


def _inventory_digest(names: list[str]) -> str:
    normalized = "".join(f"{name}\n" for name in sorted(names))
    return hashlib.sha256(normalized.encode()).hexdigest()


def _require_inventory(names: list[str], test: dict) -> None:
    digest = _inventory_digest(names)
    if len(names) != test["count"] or digest != test["sha256"]:
        raise RuntimeError(
            f"RCCL host-TSAN inventory changed for {test['name']}: "
            f"expected count={test['count']}, sha256={test['sha256']}; "
            f"got count={len(names)}, sha256={digest}"
        )


def _gtest_filter(excluded: tuple[str, ...]) -> tuple[str, ...]:
    return (f"--gtest_filter=-{':'.join(excluded)}",) if excluded else ()


def _run_gtest(bin_dir: Path, test: dict, base_env: dict[str, str]) -> None:
    executable = bin_dir / test["name"]
    require_direct_clang_tsan(executable, base_env)

    listed = subprocess.run(
        [str(executable), "--gtest_list_tests"],
        capture_output=True,
        text=True,
        env=base_env,
        check=True,
    )
    full_inventory = parse_gtest_listed_tests(listed.stdout)
    _require_inventory(full_inventory, test)

    excluded = tuple(test.get("exclude", ()))
    arguments = _gtest_filter(excluded)
    env = base_env.copy()
    if excluded:
        missing = set(excluded) - set(full_inventory)
        if missing:
            raise RuntimeError(
                f"RCCL host-TSAN exclusion inventory changed for {test['name']}: "
                f"missing {sorted(missing)}"
            )
        selected = subprocess.run(
            [str(executable), *arguments, "--gtest_list_tests"],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        selected_names = parse_gtest_listed_tests(selected.stdout)
        expected_names = set(full_inventory) - set(excluded)
        if (
            set(selected_names) != expected_names
            or len(selected_names) != test["executed"]
        ):
            raise RuntimeError(
                f"RCCL host-TSAN filtered inventory changed for {test['name']}: "
                f"expected {test['executed']} exact tests; got {len(selected_names)}"
            )
        # These death tests intentionally fault and cannot execute under TSAN.
        env["TSAN_OPTIONS"] += ":allocator_may_return_null=1"

    command = [
        str(executable),
        *arguments,
        "--gtest_brief=1",
        "--gtest_color=no",
    ]
    print(f"++ Exec {shlex.join(command)}", flush=True)
    result = subprocess.run(
        command,
        cwd=bin_dir,
        capture_output=True,
        text=True,
        env=env,
    )
    output = result.stdout + result.stderr
    sys.stdout.write(output)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    require_gtest_execution(output)
    _require_pass_summary(output, test["executed"], test["name"])


def _require_pass_summary(output: str, expected_count: int, name: str) -> None:
    """Require one final GoogleTest summary and reject child-process summaries."""
    summaries = [int(value) for value in _PASSED_RE.findall(output)]
    if summaries != [expected_count]:
        raise RuntimeError(
            f"unexpected RCCL pass summary for {name}: "
            f"expected [{expected_count}], got {summaries}"
        )


def _run_net_telemetry(bin_dir: Path, env: dict[str, str]) -> None:
    executable = bin_dir / NET_TELEMETRY_BINARY
    require_direct_clang_tsan(executable, env)
    command = [str(executable)]
    print(f"++ Exec {shlex.join(command)}", flush=True)
    result = subprocess.run(
        command,
        cwd=bin_dir,
        capture_output=True,
        text=True,
        env=env,
    )
    output = result.stdout + result.stderr
    sys.stdout.write(output)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    lines = tuple(line.strip() for line in output.splitlines())
    observed = tuple(line for line in lines if line in NET_TELEMETRY_CASES)
    if observed != NET_TELEMETRY_CASES:
        raise RuntimeError(
            "RCCL NetTelemetry host-TSAN inventory changed: "
            f"expected {len(NET_TELEMETRY_CASES)} ordered cases; got {observed}"
        )
    if lines.count("ALL PASSED") != 1 or any("FAIL:" in line for line in lines):
        raise RuntimeError(
            "RCCL NetTelemetry did not report one exact ALL PASSED result"
        )


def main() -> int:
    try:
        require_no_gpu_nodes()
        bin_dir = Path(os.environ["THEROCK_BIN_DIR"]).resolve()
        env = native_host_tsan_environment()
        for test in GTEST_INVENTORIES:
            _run_gtest(bin_dir, test, env)
        _run_net_telemetry(bin_dir, env)
        return 0
    except (KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
