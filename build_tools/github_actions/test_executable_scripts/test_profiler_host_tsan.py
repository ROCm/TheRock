#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run exact CPU-only profiler GoogleTest inventories under host TSAN."""

import hashlib
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from host_tsan_instrumentation import (
    native_host_tsan_environment,
    parse_gtest_listed_tests,
    require_direct_clang_tsan,
    require_gtest_execution,
    require_no_gpu_nodes,
)


COMPONENTS = {
    "rocprofiler-compute": (
        {
            "path": "libexec/rocprofiler-compute/tests/test-rocprofiler-compute-tool",
            "inventory_count": 79,
            "inventory_sha256": (
                "e7a0c090bec082c7701f1c15da2f1616eddbde64b8784df1b6376f4eb8a2d868"
            ),
            # Six DISABLED_ cases are listed by GoogleTest but not executed.
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
            "passed": 33,
            "skipped": 0,
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
    # This is the native, mock-backed unit binary only. The ordinary
    # rocprofiler-systems CTest/pytest payload launches profiling integration
    # workloads and is intentionally excluded from the device-free lane.
    "rocprofiler-systems": (
        {
            "path": (
                "share/rocprofiler-systems/tests/unit-tests/bin/"
                "rocprof-sys-unit-tests"
            ),
            "inventory_count": 1452,
            "inventory_sha256": (
                "eca8383c653a46e316a6452b914f726bcbd67353e1e2181625871e0f7540caa2"
            ),
            "passed": 1452,
            "skipped": 0,
        },
    ),
}

_PASSED_RE = re.compile(r"\[\s*PASSED\s*\]\s+(\d+) tests?\.")
_SKIPPED_RE = re.compile(r"\[\s*SKIPPED\s*\]\s+(\d+) tests?")
_SKIPPED_NAME_RE = re.compile(r"^\[\s*SKIPPED\s*\]\s+(\S+)\s*$", re.MULTILINE)


def _normalize_names(names: list[str]) -> str:
    return "".join(f"{name}\n" for name in sorted(names))


def _validate_inventory(executable: Path, names: list[str], expected: dict) -> None:
    normalized = _normalize_names(names)
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    if (
        len(names) != expected["inventory_count"]
        or digest != expected["inventory_sha256"]
    ):
        raise RuntimeError(
            f"{executable.name} host-TSAN inventory changed before execution: "
            f"expected count={expected['inventory_count']}, "
            f"sha256={expected['inventory_sha256']}; "
            f"got count={len(names)}, sha256={digest}"
        )


def _validate_result(executable: Path, output: str, expected: dict) -> None:
    if expected["skipped"] == 0:
        require_gtest_execution(output)
    elif re.search(r"\bRunning 0 tests? from 0 test suites?\b", output):
        raise RuntimeError("host-TSAN GoogleTest execution ran zero tests")
    passed_match = _PASSED_RE.search(output)
    skipped_match = _SKIPPED_RE.search(output)
    passed = int(passed_match.group(1)) if passed_match else 0
    skipped = int(skipped_match.group(1)) if skipped_match else 0
    if passed != expected["passed"] or skipped != expected["skipped"]:
        raise RuntimeError(
            f"unexpected GoogleTest result for {executable.name}: "
            f"expected passed={expected['passed']}, skipped={expected['skipped']}; "
            f"got passed={passed}, skipped={skipped}"
        )
    skipped_names = set(_SKIPPED_NAME_RE.findall(output))
    expected_skipped_names = set(expected.get("skipped_names", ()))
    if skipped_names != expected_skipped_names:
        raise RuntimeError(
            f"unexpected skipped tests for {executable.name}: "
            f"expected {sorted(expected_skipped_names)}; got {sorted(skipped_names)}"
        )


def _test_environment(prefix: Path, component: str) -> dict[str, str]:
    env = native_host_tsan_environment()
    library_dirs = (
        prefix / "lib",
        prefix / "lib" / "rocm_sysdeps" / "lib",
        prefix / "lib" / "llvm" / "lib",
        prefix / "lib" / "rocprofiler-compute",
        prefix / "lib" / "rocprofiler-systems",
    )
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [*(str(path) for path in library_dirs), env.get("LD_LIBRARY_PATH", "")]
    ).rstrip(os.pathsep)
    if component == "rocprofiler-sdk":
        env["ROCPROFILER_METRICS_PATH"] = str(prefix / "share" / "rocprofiler-sdk")
    return env


def _run_test_binary(command: list[str], env: dict[str, str], component: str):
    """Run a profiler test binary with any required cwd isolation."""
    run_args = {
        "capture_output": True,
        "text": True,
        "env": env,
    }
    if component == "rocprofiler-systems":
        # The unit suite creates relative *.bin fixtures. The matrix mounts the
        # TheRock checkout read-only, so never inherit its working directory.
        with tempfile.TemporaryDirectory(
            prefix="therock-rocprofiler-systems-"
        ) as test_cwd:
            run_args["env"] = {**env, "PWD": test_cwd}
            return subprocess.run(command, cwd=test_cwd, **run_args)
    return subprocess.run(command, **run_args)


def main() -> int:
    try:
        component = os.environ["TEST_COMPONENT"]
        try:
            tests = COMPONENTS[component]
        except KeyError as error:
            raise RuntimeError(
                f"unsupported profiler host-TSAN component: {component}"
            ) from error

        require_no_gpu_nodes()
        prefix = Path(os.environ["THEROCK_BIN_DIR"]).resolve().parent
        env = _test_environment(prefix, component)
        for expected in tests:
            executable = prefix / expected["path"]
            require_direct_clang_tsan(executable, env)
            listed = subprocess.run(
                [str(executable), "--gtest_list_tests"],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
            names = parse_gtest_listed_tests(listed.stdout)
            _validate_inventory(executable, names, expected)

            command = [str(executable)]
            print(f"++ Exec {shlex.join(command)}", flush=True)
            result = _run_test_binary(command, env, component)
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
            if result.returncode:
                raise subprocess.CalledProcessError(result.returncode, command)
            _validate_result(executable, result.stdout + result.stderr, expected)
        return 0
    except (KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
