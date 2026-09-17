#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run rocRoller's positive CPU-only host-TSAN GoogleTest selection."""

import hashlib
import os
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


HOST_SAFE_FILTERS = (
    "ErrorFixtureDeathTest.*",
    "ArgumentLoaderTest.*",
    "AssemblerTest.*",
    "ControlGraphTest.*",
    "CommandTest.Basic",
    "CommandTest.ToString",
    "CommandTest.ConvertOp",
    "CommandTest.VectorAdd",
    "CommandTest.XopInputOutputs",
    "CommandTest.BlockScaleInline",
    "CommandTest.BlockScaleSeparate",
    "CommandTest.SetCommandArguments",
    "CommandTest.FindCommandArguments",
    "CommandTest.GetRuntimeArguments",
    "CommandTest.CommandKernelPredicates",
    "ComponentTest.*",
)
# This test contains no executable assertions in Release builds: its body calls
# GTEST_SKIP() when NDEBUG is defined. Keep the exclusion explicit and pinned;
# require_gtest_execution() still rejects any skip from the admitted inventory.
RELEASE_MODE_NON_EXECUTING_TESTS = ("CommandTest.DuplicateOp",)
EXPECTED_CANDIDATE_INVENTORY_COUNT = 22
EXPECTED_CANDIDATE_INVENTORY_SHA256 = (
    "5d5e8f7c345bd81937d9fcf5884609f42851cc6bd2e88fb8d3024094aa065dce"
)
EXPECTED_INVENTORY_COUNT = 21
EXPECTED_INVENTORY_SHA256 = (
    "cddec42ec6529f60bfe2ab80251974682489a0aad3003426d92bca4462419ca6"
)


def _count_listed_tests(output: str) -> int:
    return len(parse_gtest_listed_tests(output))


def _validate_inventory(output: str) -> int:
    names = sorted(parse_gtest_listed_tests(output))
    digest = hashlib.sha256(
        "".join(f"{name}\n" for name in names).encode()
    ).hexdigest()
    if len(names) != EXPECTED_INVENTORY_COUNT or digest != EXPECTED_INVENTORY_SHA256:
        raise RuntimeError(
            "rocRoller host-TSAN inventory changed before execution: "
            f"expected count={EXPECTED_INVENTORY_COUNT}, "
            f"sha256={EXPECTED_INVENTORY_SHA256}; "
            f"got count={len(names)}, sha256={digest}"
        )
    return len(names)


def _validate_candidate_inventory(output: str) -> None:
    names = sorted(parse_gtest_listed_tests(output))
    digest = hashlib.sha256(
        "".join(f"{name}\n" for name in names).encode()
    ).hexdigest()
    excluded = set(RELEASE_MODE_NON_EXECUTING_TESTS)
    if (
        len(names) != EXPECTED_CANDIDATE_INVENTORY_COUNT
        or digest != EXPECTED_CANDIDATE_INVENTORY_SHA256
        or not excluded.issubset(names)
    ):
        raise RuntimeError(
            "rocRoller host-TSAN candidate inventory changed before execution: "
            f"expected count={EXPECTED_CANDIDATE_INVENTORY_COUNT}, "
            f"sha256={EXPECTED_CANDIDATE_INVENTORY_SHA256}, "
            f"release exclusions={sorted(excluded)}; "
            f"got count={len(names)}, sha256={digest}"
        )


def main() -> int:
    try:
        require_no_gpu_nodes()
        bin_dir = Path(os.environ["THEROCK_BIN_DIR"]).resolve()
        executable = bin_dir / "rocroller-tests"
        env = native_host_tsan_environment()
        require_direct_clang_tsan(executable, env)
        candidate_filter_arg = "--gtest_filter=" + ":".join(
            (*HOST_SAFE_FILTERS, *RELEASE_MODE_NON_EXECUTING_TESTS)
        )
        candidates = subprocess.run(
            [str(executable), candidate_filter_arg, "--gtest_list_tests"],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        _validate_candidate_inventory(candidates.stdout)
        filter_arg = "--gtest_filter=" + ":".join(HOST_SAFE_FILTERS)
        listed = subprocess.run(
            [str(executable), filter_arg, "--gtest_list_tests"],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        count = _validate_inventory(listed.stdout)
        command = [str(executable), filter_arg]
        print(f"++ Exec {shlex.join(command)} ({count} selected tests)", flush=True)
        executed = subprocess.run(
            command,
            cwd=bin_dir,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        output = executed.stdout + executed.stderr
        print(output, end="")
        require_gtest_execution(output)
        return 0
    except (KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
