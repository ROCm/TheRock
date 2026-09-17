#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run AQLProfile's exact GPU-independent native suite under host TSAN."""

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from host_tsan_instrumentation import (
    native_host_tsan_environment,
    parse_gtest_listed_tests,
    require_direct_clang_tsan,
    require_gtest_execution,
    require_no_gpu_nodes,
)


EXPECTED_TEST_COUNT = 80
EXPECTED_INVENTORY_SHA256 = (
    "234c36c1302354dc934768f6e7705ff0fd33f1a71fcc0b7ed2160e96c26cc104"
)
EXPECTED_EXECUTABLE_COUNTS = {
    "gfx9-memory-manager-test": 7,
    "aqlprofile-test": 6,
    "command-buffer-test": 2,
    "counters-test": 3,
    "pm4-factory-test": 2,
    "logger-test": 12,
    "aql-profile-v2-test": 21,
    "aql-profile-v2-c-compatibility-test": 1,
    "command-builder-test": 3,
    "pmc-builder-test": 3,
    "gfx9-command-builder-test": 5,
    "spm-builder-test": 4,
    "trace-config-test": 7,
    "sqtt-builder-test": 4,
    # Required and linkage-audited, but every hardware-initializing case is excluded.
    "utility_tests": 0,
}
EXPECTED_EXCLUSIONS = {
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
_ENTRY_KEYS = {"name", "kind", "tests"}
_ENTRY_KINDS = {"gtest", "standalone"}

_PASSED_RE = re.compile(r"\[\s*PASSED\s*\]\s+(\d+) tests?\.")


def _normalized_inventory(entries: list[dict[str, Any]]) -> str:
    rows = sorted(
        f"{entry['name']}\t{test}"
        for entry in entries
        for test in entry.get("tests", [])
    )
    return "".join(f"{row}\n" for row in rows)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"required AQLProfile host-only manifest is missing: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid AQLProfile host-only manifest: {error}") from error
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported AQLProfile host-only manifest schema")
    entries = manifest.get("executables")
    if not isinstance(entries, list):
        raise RuntimeError("AQLProfile host-only manifest has no executable inventory")

    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != _ENTRY_KEYS:
            raise RuntimeError(
                "AQLProfile host-only manifest entry must contain exactly "
                f"{sorted(_ENTRY_KEYS)}"
            )
        if not isinstance(entry["name"], str) or not entry["name"]:
            raise RuntimeError("AQLProfile host-only manifest entry has invalid name")
        if (
            not isinstance(entry["kind"], str)
            or entry["kind"] not in _ENTRY_KINDS
        ):
            raise RuntimeError(
                "AQLProfile host-only manifest entry has unsupported kind: "
                f"{entry['kind']!r}"
            )

    names = [entry["name"] for entry in entries]
    if (
        set(names) != set(EXPECTED_EXECUTABLE_COUNTS)
        or len(names) != len(set(names))
    ):
        raise RuntimeError("AQLProfile host-only executable inventory changed")
    for entry in entries:
        tests = entry.get("tests")
        if not isinstance(tests, list) or not all(
            isinstance(test, str) and test and not set(test) & {"*", "?", ":"}
            for test in tests
        ):
            raise RuntimeError(f"invalid exact test selector for {entry.get('name')}")
        if len(tests) != EXPECTED_EXECUTABLE_COUNTS[entry["name"]]:
            raise RuntimeError(
                f"AQLProfile per-binary inventory changed: {entry['name']}"
            )
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
    try:
        exclusions = {key: set(value) for key, value in manifest["excluded"].items()}
    except (KeyError, AttributeError, TypeError):
        exclusions = {}
    if exclusions != EXPECTED_EXCLUSIONS:
        raise RuntimeError("AQLProfile host-only exclusion inventory changed")

    normalized = _normalized_inventory(entries)
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    count = normalized.count("\n")
    if count != EXPECTED_TEST_COUNT or digest != EXPECTED_INVENTORY_SHA256:
        raise RuntimeError(
            "AQLProfile host-TSAN positive inventory changed before execution: "
            f"expected count={EXPECTED_TEST_COUNT}, "
            f"sha256={EXPECTED_INVENTORY_SHA256}; "
            f"got count={count}, sha256={digest}"
        )
    return entries


def _run_gtest(executable: Path, names: list[str], env: dict[str, str]) -> None:
    selector = ":".join(names)
    listed = subprocess.run(
        [str(executable), f"--gtest_filter={selector}", "--gtest_list_tests"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    selected = parse_gtest_listed_tests(listed.stdout)
    if len(selected) != len(names) or set(selected) != set(names):
        raise RuntimeError(
            f"AQLProfile selector inventory mismatch for {executable.name}: "
            f"expected={sorted(names)}, got={sorted(selected)}"
        )
    command = [str(executable), f"--gtest_filter={selector}"]
    print(f"++ Exec {shlex.join(command)}", flush=True)
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    output = result.stdout + result.stderr
    require_gtest_execution(output)
    summaries = [int(value) for value in _PASSED_RE.findall(output)]
    if summaries != [len(names)]:
        raise RuntimeError(
            f"incorrect GoogleTest pass summary for {executable.name}: "
            f"expected {len(names)}, got {summaries}"
        )


def main() -> int:
    try:
        require_no_gpu_nodes()
        prefix = Path(os.environ["THEROCK_BIN_DIR"]).resolve().parent
        root = prefix / "share" / "hsa-amd-aqlprofile" / "tests" / "host-tsan"
        entries = _load_manifest(root / "host_tsan_tests.json")
        env = native_host_tsan_environment()
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(prefix / "lib"), env.get("LD_LIBRARY_PATH", "")]
        ).rstrip(os.pathsep)
        executables = {
            entry["name"]: root / "bin" / entry["name"] for entry in entries
        }
        for executable in executables.values():
            require_direct_clang_tsan(executable, env)
        for entry in entries:
            if not entry["tests"]:
                continue
            executable = executables[entry["name"]]
            if entry.get("kind") == "standalone":
                command = [str(executable)]
                print(f"++ Exec {shlex.join(command)}", flush=True)
                subprocess.run(command, env=env, check=True)
            else:
                _run_gtest(executable, entry["tests"], env)
        print(
            f"AQLProfile host-TSAN passed {EXPECTED_TEST_COUNT} exact cases "
            f"(sha256={EXPECTED_INVENTORY_SHA256})"
        )
        return 0
    except (KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
