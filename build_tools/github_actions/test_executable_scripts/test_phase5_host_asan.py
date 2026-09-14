#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run the fail-closed Phase 5 CPU-only host-ASAN selectors.

Every selector validates its complete test inventory before execution. This is
intentional: adding, removing, or renaming a test requires an explicit review
of the expected count and digest before that test can enter the host-only lane.
"""

import hashlib
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

GTEST_COMPONENTS = {
    "rocprofiler-compute": (
        {
            "path": "libexec/rocprofiler-compute/tests/test-rocprofiler-compute-tool",
            "inventory_count": 79,
            "inventory_sha256": (
                "e7a0c090bec082c7701f1c15da2f1616eddbde64b8784df1b6376f4eb8a2d868"
            ),
            "passed": 73,
            "skipped": 0,
        },
        {
            "path": "libexec/rocprofiler-compute/tests/test-pc-sampling-collector",
            "inventory_count": 59,
            "inventory_sha256": (
                "99622a98a663fa922e1348532ead1b1338ee078de80540425dfb9d9408884b08"
            ),
            "passed": 59,
            "skipped": 0,
        },
    ),
    "rocprofiler-sdk": (
        {
            "path": "share/rocprofiler-sdk/tests/unit-tests/bin/common-tests",
            "inventory_count": 55,
            "inventory_sha256": (
                "344faae14dd415f1b06f384593302479a70309dfd75b8bb9b42029309c3c20a5"
            ),
            "passed": 55,
            "skipped": 0,
        },
        {
            "path": "share/rocprofiler-sdk/tests/unit-tests/bin/codeobj-library-tests",
            "inventory_count": 33,
            "inventory_sha256": (
                "b85496ed53e488e5d957d64bbe7d488585f17515a72ff445e90827356344b9e5"
            ),
            "passed": 32,
            "skipped": 1,
            "skipped_names": ("codeobj_library.dwarf_matches_llvm_symbolizer",),
        },
        {
            "path": "share/rocprofiler-sdk/tests/unit-tests/bin/parser-test",
            "inventory_count": 7,
            "inventory_sha256": (
                "4f9ff40c55823f44b37ff3c4b60271a9c039498b10c43c54221093966303e1b9"
            ),
            "passed": 7,
            "skipped": 0,
        },
    ),
}

CTEST_COMPONENTS = {
    "hipfile": {
        "test_dir": "share/hipfile/test",
        "inventory_args": ("-L", "unit"),
        "run_args": ("-L", "unit"),
        "inventory_count": 668,
        "inventory_sha256": (
            "fff635ca9e088960be5f6a4942917784333cd583b9ffd1b2c0627cd063c3f864"
        ),
        "instrumented_binary": "share/hipfile/test/internal_tests",
    },
}

_ASAN_NEEDED_RE = re.compile(r"NEEDED.*libclang_rt\.asan(?:-[^.]+)?\.so")
_CTEST_NAME_RE = re.compile(r"^\s*Test\s+#\d+:\s+(.+?)\s*$")
_PASSED_RE = re.compile(r"\[\s*PASSED\s*\]\s+(\d+) tests?\.")
_SKIPPED_RE = re.compile(r"\[\s*SKIPPED\s*\]\s+(\d+) tests?")
_SKIPPED_NAME_RE = re.compile(r"^\[\s*SKIPPED\s*\]\s+(\S+)\s*$", re.MULTILINE)


def _with_option(value: str, option: str) -> str:
    return f"{value}:{option}" if value else option


def _test_environment(prefix: Path, component: str) -> dict[str, str]:
    env = os.environ.copy()
    # Preload-only execution is not host-ASAN instrumentation. Each selected
    # native binary is checked for a direct ASAN DT_NEEDED entry below.
    env.pop("LD_PRELOAD", None)
    # LeakSanitizer cannot run under ptrace. The opt-in trace mode is only for
    # collecting the separate device-node syscall proof; the normal CI mode
    # always enables leak detection.
    detect_leaks = (
        "detect_leaks=0"
        if env.get("THEROCK_HOST_ASAN_DEVICE_TRACE") == "1"
        else "detect_leaks=1"
    )
    env["ASAN_OPTIONS"] = _with_option(env.get("ASAN_OPTIONS", ""), detect_leaks)
    env["ASAN_OPTIONS"] = _with_option(env["ASAN_OPTIONS"], "halt_on_error=1")
    env["LSAN_OPTIONS"] = _with_option(env.get("LSAN_OPTIONS", ""), "exitcode=23")
    if component == "rocprofiler-sdk":
        env["ROCPROFILER_METRICS_PATH"] = str(prefix / "share/rocprofiler-sdk")
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


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _normalize_names(names: list[str]) -> str:
    return "".join(f"{name}\n" for name in sorted(names))


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


def _parse_ctest_names(output: str) -> list[str]:
    return [
        match.group(1)
        for line in output.splitlines()
        if (match := _CTEST_NAME_RE.match(line))
    ]


def _check_inventory(
    *, names: list[str], raw_output: str, expected_count: int, expected_sha256: str
) -> None:
    digest = _sha256_text(raw_output)
    if len(names) != expected_count or digest != expected_sha256:
        raise RuntimeError(
            "host-ASAN inventory changed before execution: "
            f"expected count={expected_count}, sha256={expected_sha256}; "
            f"got count={len(names)}, sha256={digest}"
        )


def _run_gtest_component(prefix: Path, component: str, env: dict[str, str]) -> None:
    for test in GTEST_COMPONENTS[component]:
        executable = prefix / test["path"]
        _require_direct_asan(executable, env)

        list_command = [str(executable), "--gtest_list_tests"]
        listed = subprocess.run(
            list_command, capture_output=True, text=True, env=env, check=True
        )
        names = _parse_gtest_names(listed.stdout)
        normalized = _normalize_names(names)
        _check_inventory(
            names=names,
            raw_output=normalized,
            expected_count=test["inventory_count"],
            expected_sha256=test["inventory_sha256"],
        )

        # This packaged GoogleTest predates --gtest_fail_if_no_test_selected.
        # The exact pre-execution count+digest check above supplies the stronger
        # fail-closed guarantee for these whole-binary selections.
        command = [str(executable)]
        print(f"++ Exec {shlex.join(command)}", flush=True)
        result = subprocess.run(command, capture_output=True, text=True, env=env)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)

        combined = result.stdout + result.stderr
        passed_match = _PASSED_RE.search(combined)
        skipped_match = _SKIPPED_RE.search(combined)
        passed = int(passed_match.group(1)) if passed_match else 0
        skipped = int(skipped_match.group(1)) if skipped_match else 0
        skipped_names = set(_SKIPPED_NAME_RE.findall(combined))
        expected_skipped_names = set(test.get("skipped_names", ()))
        if passed != test["passed"] or skipped != test["skipped"]:
            raise RuntimeError(
                f"unexpected GTest result for {executable.name}: "
                f"expected passed={test['passed']}, skipped={test['skipped']}; "
                f"got passed={passed}, skipped={skipped}"
            )
        if skipped_names != expected_skipped_names:
            raise RuntimeError(
                f"unexpected skipped tests for {executable.name}: "
                f"expected {sorted(expected_skipped_names)}; "
                f"got {sorted(skipped_names)}"
            )


def _run_ctest_component(prefix: Path, component: str, env: dict[str, str]) -> None:
    test = CTEST_COMPONENTS[component]
    test_dir = prefix / test["test_dir"]
    _require_direct_asan(prefix / test["instrumented_binary"], env)

    list_command = ["ctest", "--test-dir", str(test_dir), "-N", *test["inventory_args"]]
    listed = subprocess.run(
        list_command, capture_output=True, text=True, env=env, check=True
    )
    names = _parse_ctest_names(listed.stdout)
    normalized = _normalize_names(names)
    _check_inventory(
        names=names,
        raw_output=normalized,
        expected_count=test["inventory_count"],
        expected_sha256=test["inventory_sha256"],
    )

    command = [
        "ctest",
        "--test-dir",
        str(test_dir),
        *test["run_args"],
        "--output-on-failure",
        "--no-tests=error",
    ]
    print(f"++ Exec {shlex.join(command)}", flush=True)
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    if "***Skipped" in result.stdout or "did not run" in result.stdout:
        raise RuntimeError(f"host-ASAN CTest selector skipped a test: {component}")
    expected_summary = (
        f"100% tests passed, 0 tests failed out of {test['inventory_count']}"
    )
    if expected_summary not in result.stdout:
        raise RuntimeError(f"missing expected CTest summary: {expected_summary}")


def main() -> int:
    try:
        component = os.environ["TEST_COMPONENT"]
        if component not in GTEST_COMPONENTS and component not in CTEST_COMPONENTS:
            raise RuntimeError(f"unsupported Phase 5 host-ASAN component: {component}")
        prefix = Path(os.environ["THEROCK_BIN_DIR"]).resolve().parent
        _require_no_gpu_nodes()
        env = _test_environment(prefix, component)
        if component in GTEST_COMPONENTS:
            _run_gtest_component(prefix, component, env)
        else:
            _run_ctest_component(prefix, component, env)
        return 0
    except (KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
