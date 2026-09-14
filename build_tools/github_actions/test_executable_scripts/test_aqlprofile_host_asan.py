#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run AQLProfile's exact GPU-independent native host-ASAN slice.

The test artifact also contains legacy GPU tests and native cases that either
initialize HSA or have known leaks. This runner trusts neither broad labels nor
name patterns: it validates the installed positive manifest before listing and
executing exactly 78 cases.
"""

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from host_asan_instrumentation import (
    native_host_asan_environment,
    require_direct_clang_asan,
)

EXPECTED_TEST_COUNT = 78
EXPECTED_INVENTORY_SHA256 = (
    "f5ba616c19360c6187f80bd2baaff28ded8d731a15d1f041e71002a9ed6947f2"
)
EXPECTED_EXECUTABLES = {
    "gfx9-memory-manager-test",
    "aqlprofile-test",
    "command-buffer-test",
    "counters-test",
    "pm4-factory-test",
    "logger-test",
    "aql-profile-v2-test",
    "aql-profile-v2-c-compatibility-test",
    "command-builder-test",
    "pmc-builder-test",
    "gfx9-command-builder-test",
    "spm-builder-test",
    "trace-config-test",
    "sqtt-builder-test",
    "utility_tests",
}
EXPECTED_EXCLUSIONS = {
    "leak_sanitizer": {
        "aqlprofile-test:CountersVecTest.RegularEvents",
        "pm4-factory-test:Pm4FactoryTest.RegisterAgentAndGetAgentInfo",
    },
    "requires_hsa_hardware": {
        "utility_tests:HsaRsrcFactoryTest.FactoryCreationAndDestruction",
        "utility_tests:HsaRsrcFactoryTest.SingletonBehavior",
        "utility_tests:HsaRsrcFactoryTest.CpuAgentCountNonZero",
        "utility_tests:HsaRsrcFactoryTest.GpuAgentCountValid",
        "utility_tests:HsaRsrcFactoryTest.GetCpuAgentInfoReturnsValid",
        "utility_tests:HsaRsrcFactoryTest.GetCpuAgentInfoOutOfRange",
        "utility_tests:HsaRsrcFactoryTest.GetGpuAgentInfoOutOfRange",
    },
    "disabled": {
        "sqtt-builder-test:SqttBuilderTest.DISABLED_BufferStepCalculation",
    },
    "stale_registration": {"testv2"},
}

_PASSED_RE = re.compile(r"\[\s*PASSED\s*\]\s+(\d+) tests?\.")
_DEVICE_NODE_RE = re.compile(r"/dev/(?:kfd|dri)(?:/|\b)")


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
    library_entries = [str(prefix / "lib")]
    if env.get("LD_LIBRARY_PATH"):
        library_entries.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(library_entries)
    return env


def _require_no_gpu_nodes() -> None:
    present = [path for path in (Path("/dev/kfd"), Path("/dev/dri")) if path.exists()]
    if present:
        raise RuntimeError(f"host-only runner unexpectedly exposes GPU nodes: {present}")


def _normalized_inventory(entries: list[dict[str, Any]]) -> str:
    rows = sorted(
        f"{entry['name']}\t{test}"
        for entry in entries
        for test in entry.get("tests", [])
    )
    return "".join(f"{row}\n" for row in rows)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"required AQLProfile host-ASAN manifest is missing: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid AQLProfile host-ASAN manifest: {error}") from error

    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported AQLProfile host-ASAN manifest schema")
    entries = manifest.get("executables")
    if not isinstance(entries, list):
        raise RuntimeError("AQLProfile host-ASAN manifest has no executable inventory")

    names = [entry.get("name") for entry in entries if isinstance(entry, dict)]
    if len(entries) != len(EXPECTED_EXECUTABLES) or set(names) != EXPECTED_EXECUTABLES:
        raise RuntimeError(
            "AQLProfile host-ASAN executable inventory changed: "
            f"expected={sorted(EXPECTED_EXECUTABLES)}, got={sorted(map(str, names))}"
        )
    if len(names) != len(set(names)):
        raise RuntimeError("AQLProfile host-ASAN executable inventory has duplicates")

    for entry in entries:
        tests = entry.get("tests")
        if not isinstance(tests, list) or not all(
            isinstance(test, str) and test and not set(test) & {"*", "?", ":"}
            for test in tests
        ):
            raise RuntimeError(f"invalid exact test selector for {entry.get('name')}")
        if len(tests) != len(set(tests)):
            raise RuntimeError(f"duplicate test selector for {entry['name']}")

    standalone = [entry for entry in entries if entry.get("kind") == "standalone"]
    if standalone != [
        {
            "name": "aql-profile-v2-c-compatibility-test",
            "kind": "standalone",
            "tests": ["aql-profile-v2-c-compatibility-test"],
        }
    ]:
        raise RuntimeError("AQLProfile standalone-test inventory changed")
    empty = {entry["name"] for entry in entries if not entry["tests"]}
    if empty != {"utility_tests"}:
        raise RuntimeError(f"unexpected empty AQLProfile selection: {sorted(empty)}")

    exclusions = manifest.get("excluded")
    try:
        normalized_exclusions = {
            key: set(value) for key, value in exclusions.items()
        }
    except (AttributeError, TypeError):
        normalized_exclusions = {}
    if normalized_exclusions != EXPECTED_EXCLUSIONS:
        raise RuntimeError("AQLProfile host-ASAN exclusion inventory changed")

    normalized = _normalized_inventory(entries)
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    count = normalized.count("\n")
    if count != EXPECTED_TEST_COUNT or digest != EXPECTED_INVENTORY_SHA256:
        raise RuntimeError(
            "AQLProfile host-ASAN positive inventory changed before execution: "
            f"expected count={EXPECTED_TEST_COUNT}, sha256={EXPECTED_INVENTORY_SHA256}; "
            f"got count={count}, sha256={digest}"
        )
    return entries


def _parse_gtest_names(output: str) -> list[str]:
    names: list[str] = []
    suite = ""
    for line in output.splitlines():
        if line and not line[0].isspace() and line.endswith("."):
            suite = line[:-1]
        elif suite and line.startswith("  ") and not line.startswith("    "):
            test = line.strip().split("  #", 1)[0]
            if test:
                names.append(f"{suite}.{test}")
    return names


def _check_trace(trace: str, command: list[str]) -> None:
    if match := _DEVICE_NODE_RE.search(trace):
        raise RuntimeError(
            f"GPU device-node access {match.group(0)!r} from {shlex.join(command)}"
        )


def _execute(
    command: list[str], env: dict[str, str], *, trace_mode: bool
) -> subprocess.CompletedProcess[str]:
    if not trace_mode:
        return subprocess.run(command, capture_output=True, text=True, env=env)

    with tempfile.NamedTemporaryFile(prefix="aqlprofile-host-asan-", suffix=".strace") as trace_file:
        traced_command = [
            "strace",
            "-f",
            "-qq",
            "-e",
            "trace=open,openat,openat2",
            "-o",
            trace_file.name,
            "--",
            *command,
        ]
        result = subprocess.run(
            traced_command, capture_output=True, text=True, env=env
        )
        trace_file.seek(0)
        _check_trace(trace_file.read().decode(errors="replace"), command)
        return result


def _run_gtest(
    executable: Path,
    expected_names: list[str],
    env: dict[str, str],
    *,
    trace_mode: bool,
) -> None:
    selector = ":".join(expected_names)
    list_command = [
        str(executable),
        f"--gtest_filter={selector}",
        "--gtest_list_tests",
    ]
    listed = _execute(list_command, env, trace_mode=trace_mode)
    if listed.returncode:
        raise subprocess.CalledProcessError(listed.returncode, list_command)
    listed_names = _parse_gtest_names(listed.stdout)
    if (
        len(listed_names) != len(expected_names)
        or len(set(listed_names)) != len(expected_names)
        or set(listed_names) != set(expected_names)
    ):
        raise RuntimeError(
            f"AQLProfile selector inventory mismatch for {executable.name}: "
            f"expected={sorted(expected_names)}, got={sorted(listed_names)}"
        )

    command = [
        str(executable),
        f"--gtest_filter={selector}",
    ]
    print(f"++ Exec {shlex.join(command)}", flush=True)
    result = _execute(command, env, trace_mode=trace_mode)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    summaries = [int(value) for value in _PASSED_RE.findall(result.stdout + result.stderr)]
    if summaries != [len(expected_names)]:
        raise RuntimeError(
            f"missing or incorrect GoogleTest pass summary for {executable.name}: "
            f"expected {len(expected_names)}, got {summaries}"
        )


def run(prefix: Path, env: dict[str, str]) -> None:
    root = prefix / "share" / "hsa-amd-aqlprofile" / "tests" / "host-asan"
    entries = _load_manifest(root / "host_asan_tests.json")
    bin_dir = root / "bin"
    executables = {entry["name"]: bin_dir / entry["name"] for entry in entries}

    # All 15 binaries are package requirements, including utility_tests whose
    # seven cases are explicitly excluded because they initialize HSA.
    for executable in executables.values():
        require_direct_clang_asan(executable, env)

    trace_mode = env.get("THEROCK_HOST_ASAN_DEVICE_TRACE") == "1"
    for entry in entries:
        if not entry["tests"]:
            continue
        executable = executables[entry["name"]]
        if entry.get("kind") == "standalone":
            command = [str(executable)]
            print(f"++ Exec {shlex.join(command)}", flush=True)
            result = _execute(command, env, trace_mode=trace_mode)
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
            if result.returncode:
                raise subprocess.CalledProcessError(result.returncode, command)
        else:
            _run_gtest(
                executable,
                entry["tests"],
                env,
                trace_mode=trace_mode,
            )


def main() -> int:
    try:
        prefix = Path(os.environ["THEROCK_BIN_DIR"]).resolve().parent
        _require_no_gpu_nodes()
        env = _test_environment(prefix)
        run(prefix, env)
        if env.get("THEROCK_HOST_ASAN_DEVICE_TRACE") == "1":
            print(
                "AQLProfile host-ASAN trace observed zero /dev/kfd or /dev/dri "
                "accesses"
            )
        print(
            f"AQLProfile host-ASAN passed {EXPECTED_TEST_COUNT} exact cases "
            f"(sha256={EXPECTED_INVENTORY_SHA256})"
        )
        return 0
    except (KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
