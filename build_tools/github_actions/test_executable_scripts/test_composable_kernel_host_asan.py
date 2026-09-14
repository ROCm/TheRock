#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run only Composable Kernel executables proven independent of a GPU device."""

import hashlib
import logging
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

COMPOSABLE_KERNEL_HOST_TESTS = {
    "unit_sequence": (
        89,
        "48b0180ac8dd4058e121b00019d41cc75620f5762d141182378320cc8c8c1cec",
    ),
    "unit_ford": (
        30,
        "b4fcd289312a3951db8fa4068e5f264abff0d7846ec236e53aecc512262f5d0d",
    ),
    "unit_sequence_helper": (
        18,
        "8552f22d4ffe5bac62d7928a70be916190ecff536ca2a9871704febf284f252d",
    ),
    "unit_tensor_descriptor_functors": (
        19,
        "cd936d51cb8abfa0735af99158418916562965ad5a45b70b7c2ce54b11cbc9cf",
    ),
    "unit_index_expression": (
        3,
        "866dc0c2ade961f8fa63bdab1dcedc335f63fa611628f1caf370cfb3bfe67563",
    ),
    "test_ck_tile_tuple_apply": (
        17,
        "e5a5321f5d97360ab96f5a7801c3b78566c99629bc98d20c1ac29e82ce9c9bf9",
    ),
    "ck_tile_unit_sequence": (
        88,
        "b14c92484bbcb0dc925de2fca8706be962cc459f0705e4c148e7dc56153dd432",
    ),
    "test_ck_tile_sequence": (
        67,
        "ee2fb5fc903c4086f3fda6df67bc85d1afb920c1a7f917c9eb696ad270d1bbd8",
    ),
    "test_ck_tile_static_ford": (
        13,
        "073838fe23306321c61e29f0cf8c463d9deaf1658700ebb68bec46703dbee8d3",
    ),
    "test_print_sequence": (
        3,
        "4da67037a2c76ebb39a63c82eac43c2926bb680cfecae00f7d1c87af77f9451e",
    ),
    "test_print_array": (
        4,
        "aba725e923e0f9e6d21f6e152788da6a0220be99f24dd38b69ffd55ab7e284d4",
    ),
    "test_print_tuple": (
        4,
        "e19abe39283f875a57ced33f2f5812c58c2653725c041115c42fe5cd40cc4f85",
    ),
    "test_print_coordinate_transform": (
        5,
        "257d936dd2ec411e43c2ceda69bb623ae947040df5cc1d1a2bf55ae22835fa64",
    ),
    "test_print_static_encoding_pattern": (
        6,
        "392b9565c293f73af54b9ef3ead9941487bcede404729ef4e781ad368ef71b59",
    ),
    "test_print_buffer_view": (
        4,
        "6a2cf05e7f3ee878674ac0e63cada292d940ce12c6dd06274a8355b174923910",
    ),
    "test_print_basic_types": (
        6,
        "155a8a5b9659fa1153a8d559f1e12e599e64bbddc69cc78b4b9d7a1a918c65bb",
    ),
    "test_custom_type": (
        31,
        "5638d38cef93e058a8f6820dc8ad3380471c2953116adc4966872262aa39a67d",
    ),
    "test_type_convert_const": (
        2,
        "25e397ec22e9bafa51bfb176bf3b9605a719ddd4938130ef7b878643d8c9e0d3",
    ),
}

_ASAN_NEEDED_RE = re.compile(r"NEEDED.*libclang_rt\.asan(?:-[^.]+)?\.so")
_PASSED_RE = re.compile(r"\[\s*PASSED\s*\]\s+(\d+) tests?\.")


def _with_option(value: str, option: str) -> str:
    return f"{value}:{option}" if value else option


def _test_environment() -> dict[str, str]:
    env = os.environ.copy()
    # A preload can make an uninstrumented binary appear to run under ASAN.
    # The direct DT_NEEDED check below is the admission invariant.
    env.pop("LD_PRELOAD", None)
    detect_leaks = (
        "detect_leaks=0"
        if env.get("THEROCK_HOST_ASAN_DEVICE_TRACE") == "1"
        else "detect_leaks=1"
    )
    env["ASAN_OPTIONS"] = _with_option(env.get("ASAN_OPTIONS", ""), detect_leaks)
    env["ASAN_OPTIONS"] = _with_option(env["ASAN_OPTIONS"], "halt_on_error=1")
    env["LSAN_OPTIONS"] = _with_option(env.get("LSAN_OPTIONS", ""), "exitcode=23")
    return env


def _require_no_gpu_nodes() -> None:
    present = [path for path in (Path("/dev/kfd"), Path("/dev/dri")) if path.exists()]
    if present:
        raise RuntimeError(
            f"host-only runner unexpectedly exposes GPU nodes: {present}"
        )


def _require_direct_asan(executable: Path, env: dict[str, str]) -> None:
    if not executable.is_file():
        raise RuntimeError(f"required host-ASAN executable is missing: {executable}")
    result = subprocess.run(
        ["readelf", "-d", str(executable)],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    if not _ASAN_NEEDED_RE.search(result.stdout):
        raise RuntimeError(
            f"required executable is not directly linked to Clang ASAN: {executable}"
        )


def _parse_gtest_names(output: str) -> list[str]:
    names: list[str] = []
    suite = None
    for raw_line in output.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if not raw_line[0].isspace() and raw_line.rstrip().endswith("."):
            suite = raw_line.split("#", 1)[0].strip()
            continue
        if suite is not None and raw_line[0].isspace():
            case = raw_line.split("#", 1)[0].strip()
            if case:
                names.append(f"{suite}{case}")
    return names


def _validate_inventory(
    executable: Path, names: list[str], expected_count: int, expected_sha256: str
) -> None:
    normalized = "".join(f"{name}\n" for name in sorted(names))
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    if len(names) != expected_count or digest != expected_sha256:
        raise RuntimeError(
            f"host-ASAN inventory changed for {executable.name}: "
            f"expected count={expected_count}, sha256={expected_sha256}; "
            f"got count={len(names)}, sha256={digest}"
        )


def _run_binary(
    executable: Path,
    expected_count: int,
    expected_sha256: str,
    bin_dir: Path,
    env: dict[str, str],
) -> None:
    _require_direct_asan(executable, env)
    listed = subprocess.run(
        [str(executable), "--gtest_list_tests"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    names = _parse_gtest_names(listed.stdout)
    _validate_inventory(executable, names, expected_count, expected_sha256)

    command = [str(executable)]
    logging.info("++ Exec %s", shlex.join(command))
    result = subprocess.run(
        command, cwd=bin_dir, capture_output=True, text=True, env=env
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    passed_match = _PASSED_RE.search(result.stdout + result.stderr)
    passed = int(passed_match.group(1)) if passed_match else 0
    if passed != expected_count:
        raise RuntimeError(
            f"unexpected GTest result for {executable.name}: "
            f"expected {expected_count} passed; got {passed}"
        )


def main() -> int:
    bin_dir = Path(os.environ["THEROCK_BIN_DIR"]).resolve()
    _require_no_gpu_nodes()
    env = _test_environment()
    for binary_name, (
        expected_count,
        expected_sha256,
    ) in COMPOSABLE_KERNEL_HOST_TESTS.items():
        executable = bin_dir / binary_name
        _run_binary(executable, expected_count, expected_sha256, bin_dir, env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
