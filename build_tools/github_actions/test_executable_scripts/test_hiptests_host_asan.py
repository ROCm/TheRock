#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run the exact GPU-independent hip-tests host-ASAN slice.

The packaged UnitDeviceTests executable also contains GPU tests.  Selection is
therefore deliberately fail closed: enumerate the positive selector first and
require its complete name inventory before executing it.
"""

import hashlib
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from host_asan_instrumentation import (
    native_host_asan_environment,
    require_direct_clang_asan,
)

EXPECTED_TEST_NAMES = (
    "Unit_hipVectorTypes_test_on_host",
    "Unit_hipTestHalfConstexpr_HostConstexpr",
    "Unit_bf16_operators_host",
    "Unit_bf162_operators_host",
    "Unit_fp16_host_operations",
    "Unit_half_isnan_host",
    "Unit_half_abs_host",
    "Unit_half_min_max_host",
    "Unit_fp8_ocp_bool_host",
    "Unit_all_fp8_ocp_vector_cvt",
    "Unit_fp8_ocp_correctness - float",
    "Unit_fp8_ocp_correctness - double",
    "Unit_fp8_ocp_vector_basic_conversions",
    "Unit_fp8_fnuz_bool_host",
    "Unit_all_fp8_fnuz_vector_cvt",
    "Unit_fp8_fnuz_correctness - float",
    "Unit_fp8_fnuz_correctness - double",
    "Unit_fp8_fnuz_vector_basic_conversions",
    "Unit__hip_cvt_e8m0_to_double_exhaustive_host",
    "Unit_ocp_fp4_from_double_full_range_host",
    "Unit_ext_ocp_fp6_roundtrip_host",
    "Unit_ext_ocp_fp6_known_values_host",
    "Unit_ext_ocp_fp6_scaled_roundtrip_host",
    "Unit_ext_ocp_fp6_fp16_bf16_source_host",
    "Unit_ext_ocp_fp8_roundtrip_host",
    "Unit_ext_ocp_fp8_known_values_host",
    "Unit_ext_ocp_fp8_scaled_decode_host",
    "Unit_ext_ocp_fp8x2_scaled_roundtrip_host",
    "Unit_ext_ocp_fp8_stochastic_rounding_host",
    "Unit_ext_ocp_fp8_fp16_bf16_source_host",
    "Unit_ext_ocp_fp4_roundtrip_host",
    "Unit_ext_ocp_fp4_known_values_host",
    "Unit_ext_ocp_fp4_scaled_roundtrip_host",
    "Unit_ext_ocp_fp4_stochastic_rounding_host",
    "Unit_ext_ocp_fp4_fp16_bf16_source_host",
)
EXPECTED_TEST_COUNT = 35
EXPECTED_ASSERTION_COUNT = 8743
EXPECTED_INVENTORY_SHA256 = (
    "3d222602c4e20ea150da38de4a8e19eafc47639911d22f77642986bb6a62ced8"
)

_LIST_HEADERS = {"All available test cases:", "Matching test cases:"}
_LIST_SUMMARY_RE = re.compile(
    r"^(\d+) (?:matching )?test cases?$", re.IGNORECASE
)
_RESULT_RE = re.compile(
    r"All tests passed\s*\(\s*(\d+) assertions? in (\d+) test cases?\s*\)",
    re.IGNORECASE,
)


def _with_option(value: str, option: str) -> str:
    return f"{value}:{option}" if value else option


def _test_environment(prefix: Path) -> dict[str, str]:
    env = native_host_asan_environment()
    trace_mode = env.get("THEROCK_HOST_ASAN_DEVICE_TRACE") == "1"
    env["ASAN_OPTIONS"] = _with_option(
        env.get("ASAN_OPTIONS", ""),
        "detect_leaks=0" if trace_mode else "detect_leaks=1",
    )
    env["ASAN_OPTIONS"] = _with_option(env["ASAN_OPTIONS"], "halt_on_error=1")
    env["LSAN_OPTIONS"] = _with_option(env.get("LSAN_OPTIONS", ""), "exitcode=23")
    env["ROCM_PATH"] = str(prefix)
    library_path = str(prefix / "lib")
    if env.get("LD_LIBRARY_PATH"):
        library_path = f"{library_path}:{env['LD_LIBRARY_PATH']}"
    env["LD_LIBRARY_PATH"] = library_path
    return env


def _require_no_gpu_nodes() -> None:
    present = [path for path in (Path("/dev/kfd"), Path("/dev/dri")) if path.exists()]
    if present:
        raise RuntimeError(
            f"host-only runner unexpectedly exposes GPU nodes: {present}"
        )


def _normalize_names(names: list[str] | tuple[str, ...]) -> str:
    return "".join(f"{name}\n" for name in sorted(names))


def _inventory_sha256(names: list[str] | tuple[str, ...]) -> str:
    return hashlib.sha256(_normalize_names(names).encode()).hexdigest()


def _parse_catch_list(output: str) -> tuple[list[str], int]:
    """Parse Catch2's console listing without accepting wrapped/unknown lines."""
    names: list[str] = []
    in_list = False
    reported_count = -1
    for line in output.splitlines():
        stripped = line.strip()
        if stripped in _LIST_HEADERS:
            in_list = True
            continue
        if not in_list:
            continue
        if match := _LIST_SUMMARY_RE.fullmatch(stripped):
            reported_count = int(match.group(1))
            break
        # Catch names use two spaces; tags and wrapped descriptions use deeper
        # indentation.  All admitted names fit on one line.
        if line.startswith("  ") and not line.startswith("    ") and stripped:
            names.append(stripped)
    return names, reported_count


def _check_inventory(names: list[str], reported_count: int) -> None:
    digest = _inventory_sha256(names)
    expected_names = set(EXPECTED_TEST_NAMES)
    if (
        reported_count != EXPECTED_TEST_COUNT
        or len(names) != EXPECTED_TEST_COUNT
        or len(set(names)) != EXPECTED_TEST_COUNT
        or set(names) != expected_names
        or digest != EXPECTED_INVENTORY_SHA256
    ):
        raise RuntimeError(
            "hip-tests host-ASAN inventory changed before execution: "
            f"expected count={EXPECTED_TEST_COUNT}, "
            f"sha256={EXPECTED_INVENTORY_SHA256}; got reported_count="
            f"{reported_count}, parsed_count={len(names)}, sha256={digest}"
        )


def _selector() -> str:
    # Catch2 commas are OR separators. None of these exact names contains a
    # comma or wildcard, so the result is a positive, exact-name selection.
    return ",".join(EXPECTED_TEST_NAMES)


def run(prefix: Path, env: dict[str, str]) -> None:
    executable = prefix / "share" / "hip" / "catch_tests" / "UnitDeviceTests"
    require_direct_clang_asan(executable, env)

    list_command = [str(executable), _selector(), "--list-tests"]
    listed = subprocess.run(list_command, capture_output=True, text=True, env=env)
    if listed.returncode:
        sys.stdout.write(listed.stdout)
        sys.stderr.write(listed.stderr)
        raise subprocess.CalledProcessError(listed.returncode, list_command)
    names, reported_count = _parse_catch_list(listed.stdout)
    _check_inventory(names, reported_count)

    command = [str(executable), _selector()]
    print(f"++ Exec {shlex.join(command)}", flush=True)
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)

    combined = result.stdout + result.stderr
    summary = _RESULT_RE.search(combined)
    if not summary:
        raise RuntimeError("missing successful Catch2 result summary")
    assertion_count, test_count = map(int, summary.groups())
    if (
        assertion_count != EXPECTED_ASSERTION_COUNT
        or test_count != EXPECTED_TEST_COUNT
    ):
        raise RuntimeError(
            "unexpected hip-tests host-ASAN result: expected "
            f"{EXPECTED_ASSERTION_COUNT} assertions in {EXPECTED_TEST_COUNT} "
            f"test cases; got {assertion_count} assertions in {test_count} test cases"
        )
    if "skipped" in combined.lower():
        raise RuntimeError("hip-tests host-ASAN selection reported a skipped case")


def main() -> int:
    try:
        prefix = Path(os.environ["THEROCK_BIN_DIR"]).resolve().parent
        _require_no_gpu_nodes()
        run(prefix, _test_environment(prefix))
        return 0
    except (KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
