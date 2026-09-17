#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run ROCgdb's exact CPU-only DWARF testsuite under host TSAN."""

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from host_tsan_instrumentation import (
    native_host_tsan_environment,
    require_direct_clang_tsan,
    require_no_gpu_nodes,
)


# Inventory at TheRock's pinned ROCgdb source commit
# 6b3d3b8b8b2995b9236b2c74d034966a42d21dd2. Only this generic CPU suite is
# admitted. gdb.rocm, GPU core-file, debug-agent and every other testsuite
# directory remain outside the host-only lane.
EXPECTED_DWARF2_COUNT = 273
EXPECTED_DWARF2_SHA256 = (
    "9bc9e7a25666589db784198e206e755cf888d2877a945d525649e93839042cb8"
)
EXPECTED_LLVM_XFAILS = (
    "gdb.dwarf2/dw2-case-insensitive.exp",
    "gdb.dwarf2/dw2-cp-infcall-ref-static.exp",
    "gdb.dwarf2/dw2-entry-value.exp",
    "gdb.dwarf2/dw2-inline-param.exp",
    "gdb.dwarf2/dw2-param-error.exp",
    "gdb.dwarf2/dw2-skip-prologue.exp",
    "gdb.dwarf2/dw2-unresolved.exp",
    "gdb.dwarf2/fission-dw-form-strx.exp",
    "gdb.dwarf2/pr13961.exp",
)
EXPECTED_USED_LLVM_XFAILS = tuple(
    name
    for name in EXPECTED_LLVM_XFAILS
    if name != "gdb.dwarf2/fission-dw-form-strx.exp"
)
EXPECTED_UNUSED_LLVM_XFAILS = ("gdb.dwarf2/fission-dw-form-strx.exp",)
EXPECTED_OUTCOME_COUNTS = {
    "GCC": {"PASS": 2715, "UNSUPPORTED": 13, "KFAIL": 6},
    "LLVM": {
        "PASS": 2331,
        "FAIL": 10,
        "UNTESTED": 12,
        "UNSUPPORTED": 38,
        "KFAIL": 4,
    },
}
OUTCOME_VALIDATION_MARKER = (
    "Validated ROCgdb CPU-only host-TSAN outcome profile: "
    "GCC pass=2715 unsupported=13 kfail=6; "
    "LLVM pass=2331 fail=10 untested=12 unsupported=38 kfail=4; "
    "approved failed files=8; unused approval files=1."
)

_SECTION_RE = re.compile(r"^-+\s+(GCC|LLVM)\s+-+$")
_COUNT_RE = re.compile(
    r"^\[[^]]+\]\s+(PASS|FAIL|UNTESTED|UNSUPPORTED|KFAIL):\s+(\d+)$"
)
_IGNORED_HEADER_RE = re.compile(r"^\[!\] Ignored Failures \(LLVM\) \((\d+)\):$")
_UNUSED_HEADER_RE = re.compile(r"^\[!\] Unused Ignored Failures \((\d+)\):$")


def _inventory_digest(names: list[str]) -> str:
    normalized = "".join(f"{name}\n" for name in sorted(names))
    return hashlib.sha256(normalized.encode()).hexdigest()


def validate_dwarf2_inventory(testsuite_dir: Path) -> list[str]:
    dwarf2_dir = testsuite_dir / "gdb.dwarf2"
    names = sorted(path.name for path in dwarf2_dir.glob("*.exp") if path.is_file())
    digest = _inventory_digest(names)
    if len(names) != EXPECTED_DWARF2_COUNT or digest != EXPECTED_DWARF2_SHA256:
        raise RuntimeError(
            "ROCgdb host-TSAN gdb.dwarf2 inventory changed before execution: "
            f"expected count={EXPECTED_DWARF2_COUNT}, "
            f"sha256={EXPECTED_DWARF2_SHA256}; "
            f"got count={len(names)}, sha256={digest}"
        )
    return names


def validate_ignore_list(path: Path) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read ROCgdb ignore list {path}: {error}") from error

    expected = {
        "Generic": [],
        "GCC": [],
        "LLVM": list(EXPECTED_LLVM_XFAILS),
    }
    if data != expected:
        raise RuntimeError(
            "ROCgdb host-TSAN expected-failure contract changed: "
            f"expected {expected}; got {data}"
        )


def _normalized_log_lines(output: str) -> list[str]:
    lines = []
    for raw_line in output.splitlines():
        if " - INFO:" in raw_line:
            raw_line = raw_line.split(" - INFO:", 1)[1]
        lines.append(raw_line.strip())
    return lines


def _parse_expected_failure_block(
    lines: list[str], header_re: re.Pattern[str]
) -> tuple[int, tuple[str, ...]]:
    headers = [
        (index, match)
        for index, line in enumerate(lines)
        if (match := header_re.fullmatch(line))
    ]
    if len(headers) != 1:
        raise RuntimeError(
            "ROCgdb host-TSAN outcome profile changed: "
            f"expected one {header_re.pattern!r} block, got {len(headers)}"
        )
    index, match = headers[0]
    names = []
    for line in lines[index + 1 :]:
        if not line:
            break
        if not line.startswith("gdb.dwarf2/"):
            break
        names.append(line)
    return int(match.group(1)), tuple(names)


def validate_outcome_profile(output: str) -> None:
    """Fail closed unless the pinned CPU-only DejaGNU outcome is unchanged."""
    lines = _normalized_log_lines(output)
    actual_counts: dict[str, dict[str, int]] = {"GCC": {}, "LLVM": {}}
    current_section = None
    for line in lines:
        section_match = _SECTION_RE.fullmatch(line)
        if section_match:
            current_section = section_match.group(1)
            continue
        if "FINAL TEST STATUS" in line:
            current_section = None
            continue
        count_match = _COUNT_RE.fullmatch(line)
        if current_section and count_match:
            status, count_text = count_match.groups()
            if status in actual_counts[current_section]:
                raise RuntimeError(
                    "ROCgdb host-TSAN outcome profile changed: duplicate "
                    f"{current_section} {status} count"
                )
            actual_counts[current_section][status] = int(count_text)

    if actual_counts != EXPECTED_OUTCOME_COUNTS:
        raise RuntimeError(
            "ROCgdb host-TSAN outcome profile changed: "
            f"expected counts={EXPECTED_OUTCOME_COUNTS}; got counts={actual_counts}"
        )

    ignored_count, ignored_names = _parse_expected_failure_block(
        lines, _IGNORED_HEADER_RE
    )
    unused_count, unused_names = _parse_expected_failure_block(
        lines, _UNUSED_HEADER_RE
    )
    if (
        ignored_count != len(ignored_names)
        or ignored_names != EXPECTED_USED_LLVM_XFAILS
    ):
        raise RuntimeError(
            "ROCgdb host-TSAN approved failure use changed: "
            f"expected {EXPECTED_USED_LLVM_XFAILS}; got {ignored_names} "
            f"with reported count {ignored_count}"
        )
    if unused_count != len(unused_names) or unused_names != EXPECTED_UNUSED_LLVM_XFAILS:
        raise RuntimeError(
            "ROCgdb host-TSAN unused failure approval changed: "
            f"expected {EXPECTED_UNUSED_LLVM_XFAILS}; got {unused_names} "
            f"with reported count {unused_count}"
        )
    if lines.count("[X] Total Failed Tests: 8") != 1:
        raise RuntimeError(
            "ROCgdb host-TSAN outcome profile changed: expected exactly eight "
            "approved failed test files"
        )
    if lines.count("[✓] GCC: PASS") != 1 or lines.count("[✓] LLVM: PASS") != 1:
        raise RuntimeError(
            "ROCgdb host-TSAN outcome profile changed: compiler status is not PASS"
        )
    if lines.count("[✓] OVERALL STATUS: PASS") != 1:
        raise RuntimeError(
            "ROCgdb host-TSAN outcome profile changed: overall status is not PASS"
        )


def run_and_capture(command: list[str], env: dict[str, str]) -> str:
    """Stream the long test log while retaining a disk-backed validation copy."""
    with tempfile.SpooledTemporaryFile(
        mode="w+", max_size=8 * 1024 * 1024, encoding="utf-8"
    ) as captured:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            captured.write(line)
        return_code = process.wait()
        captured.seek(0)
        output = captured.read()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return output


def rocgdb_host_binaries(prefix: Path) -> list[Path]:
    binaries = sorted(
        path
        for path in (prefix / "bin").glob("rocgdb-py*")
        if path.is_file()
    )
    if not binaries:
        raise RuntimeError(f"no ROCgdb host ELF binaries found below {prefix / 'bin'}")
    return binaries


def test_command(runner: Path) -> list[str]:
    return [
        sys.executable,
        str(runner),
        "--parallel",
        "-f",
        "0.25",
        # A sanitizer finding must fail its first run; retries could mask a
        # schedule-sensitive race.
        "--max-failed-retries",
        "0",
        "--tests",
        "gdb.dwarf2",
    ]


def main() -> int:
    require_no_gpu_nodes()
    try:
        prefix = Path(os.environ["THEROCK_BIN_DIR"]).resolve().parent
    except KeyError as error:
        raise RuntimeError("THEROCK_BIN_DIR is required") from error

    runner = prefix / "tests" / "rocgdb" / "test_rocgdb.py"
    testsuite_dir = prefix / "tests" / "rocgdb" / "gdb" / "testsuite"
    ignore_list = prefix / "tests" / "rocgdb" / "rocgdb_ignore_list.json"
    if not runner.is_file():
        raise RuntimeError(f"ROCgdb test runner is missing: {runner}")

    names = validate_dwarf2_inventory(testsuite_dir)
    validate_ignore_list(ignore_list)
    env = native_host_tsan_environment()
    for binary in rocgdb_host_binaries(prefix):
        require_direct_clang_tsan(binary, env)

    command = test_command(runner)
    print(
        "Validated ROCgdb CPU-only host-TSAN inventory: "
        f"{len(names)} gdb.dwarf2 .exp files; all other suites excluded."
    )
    print("Running:", shlex.join(command))
    output = run_and_capture(command, env)
    validate_outcome_profile(output)
    print(OUTCOME_VALIDATION_MARKER)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
