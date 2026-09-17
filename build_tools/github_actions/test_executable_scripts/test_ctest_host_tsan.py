#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run exact, positive CTest inventories admitted to the host-TSAN lane."""

import hashlib
import json
import os
import platform
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


COMPONENTS = {
    "origami": {
        "test_dir": "bin/origami",
        "args": ("-R", "^origami-tests$"),
        "expected_names": ("origami-tests",),
    },
    "hipfile": {
        "test_dir": "share/hipfile/test",
        "args": ("-L", "unit"),
        "inventory_count": 668,
        "inventory_sha256": (
            "fff635ca9e088960be5f6a4942917784333cd583b9ffd1b2c0627cd063c3f864"
        ),
    },
}

_CTEST_NAME_RE = re.compile(r"^\s*Test\s+#\d+:\s+(.+?)\s*$")
_GTEST_ONE_PASS_RE = re.compile(r"(?m)^\[\s*PASSED\s*\]\s+1 test\.\s*$")


def _without_aslr(*command: str) -> list[str]:
    """Return a command with ASLR disabled for Clang TSAN's fixed mappings."""
    return ["setarch", platform.machine(), "-R", *command]


def _require_one_gtest_pass(output: str, name: str) -> None:
    require_gtest_execution(output)
    if not _GTEST_ONE_PASS_RE.search(output):
        raise RuntimeError(
            f"hipfile host-TSAN test did not report one pass: {name}"
        )


def _normalize_names(names: list[str]) -> str:
    return "".join(f"{name}\n" for name in sorted(names))


def _parse_ctest_names(output: str) -> list[str]:
    return [
        match.group(1)
        for line in output.splitlines()
        if (match := _CTEST_NAME_RE.match(line))
    ]


def _validate_inventory(component: str, config: dict, names: list[str]) -> None:
    normalized = _normalize_names(names)
    if "expected_names" in config:
        expected = sorted(config["expected_names"])
        if sorted(names) != expected:
            raise RuntimeError(
                f"{component} host-TSAN inventory changed before execution: "
                f"expected {expected}; got {sorted(names)}"
            )
        return

    digest = hashlib.sha256(normalized.encode()).hexdigest()
    if len(names) != config["inventory_count"] or digest != config["inventory_sha256"]:
        raise RuntimeError(
            f"{component} host-TSAN inventory changed before execution: "
            f"expected count={config['inventory_count']}, "
            f"sha256={config['inventory_sha256']}; "
            f"got count={len(names)}, sha256={digest}"
        )


def _validate_ctest_execution(output: str, expected_count: int) -> None:
    if re.search(r"\*\*\*Skipped|Not Run|did not run", output, re.IGNORECASE):
        raise RuntimeError("host-TSAN CTest execution skipped or did not run tests")
    expected_summary = re.compile(
        rf"100% tests passed,\s+0 tests failed out of {expected_count}\b"
    )
    if not expected_summary.search(output):
        raise RuntimeError(
            "host-TSAN CTest execution did not report the exact passing inventory: "
            f"expected {expected_count} tests"
        )


def _selected_commands(test_dir: Path, args: tuple[str, ...], env: dict[str, str]):
    result = subprocess.run(
        ["ctest", "--test-dir", str(test_dir), "--show-only=json-v1", *args],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    tests = json.loads(result.stdout).get("tests", [])
    if not tests:
        raise RuntimeError("host-TSAN CTest selector resolved to zero tests")
    return tests


def _audit_direct_linkage(
    test_dir: Path, tests: list[dict], env: dict[str, str]
) -> None:
    executables = set()
    for test in tests:
        command = test.get("command") or []
        if not command:
            raise RuntimeError(
                "host-TSAN CTest entry has no command: "
                f"{test.get('name', '<unnamed>')}"
            )
        executable = Path(command[0])
        if not executable.is_absolute():
            executable = test_dir / executable
        executables.add(executable.resolve())
    for executable in sorted(executables):
        require_direct_clang_tsan(executable, env)


def _run_hipfile(
    test_dir: Path, config: dict, env: dict[str, str]
) -> None:
    """Run hipFile's frozen GoogleTest inventory one process per test.

    The installed discovery script replaces LD_LIBRARY_PATH with only the
    prefix library directories, which hides the Clang TSAN runtime before it
    can register the CTest inventory. Discover directly, then retain CTest's
    isolation guarantee by launching every selected test in its own process.
    ``cmake -E env`` keeps the binary's direct parent consistent with CTest.
    """
    executable = test_dir / "internal_tests"
    require_direct_clang_tsan(executable, env)

    listed = subprocess.run(
        [
            "cmake",
            "-E",
            "env",
            "--",
            *_without_aslr(str(executable), "--gtest_list_tests"),
        ],
        cwd=test_dir,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    names = parse_gtest_listed_tests(listed.stdout)
    _validate_inventory("hipfile", config, names)

    for index, name in enumerate(names, start=1):
        command = [
            "cmake",
            "-E",
            "env",
            "--",
            *_without_aslr(
                str(executable), f"--gtest_filter={name}", "--gtest_color=no"
            ),
        ]
        print(f"++ [{index}/{len(names)}] {shlex.join(command)}", flush=True)
        executed = subprocess.run(
            command,
            cwd=test_dir,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = executed.stdout + executed.stderr
        print(output, end="")
        _require_one_gtest_pass(output, name)

    print(f"hipfile host-TSAN: all {len(names)} isolated tests passed")


def main() -> int:
    try:
        component = os.environ["TEST_COMPONENT"]
        try:
            config = COMPONENTS[component]
        except KeyError as error:
            raise RuntimeError(
                f"unsupported CTest host-TSAN component: {component}"
            ) from error

        require_no_gpu_nodes()
        prefix = Path(os.environ["THEROCK_BIN_DIR"]).resolve().parent
        test_dir = prefix / config["test_dir"]
        args = config["args"]
        env = native_host_tsan_environment()
        library_dirs = (
            prefix / "lib",
            prefix / "lib" / "rocm_sysdeps" / "lib",
            prefix / "lib" / "llvm" / "lib",
        )
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [*(str(path) for path in library_dirs), env.get("LD_LIBRARY_PATH", "")]
        ).rstrip(os.pathsep)

        if component == "hipfile":
            _run_hipfile(test_dir, config, env)
            return 0

        listed = subprocess.run(
            ["ctest", "--test-dir", str(test_dir), "-N", *args],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        names = _parse_ctest_names(listed.stdout)
        _validate_inventory(component, config, names)
        tests = _selected_commands(test_dir, args, env)
        if sorted(test.get("name") for test in tests) != sorted(names):
            raise RuntimeError("CTest text and JSON inventories disagree")
        _audit_direct_linkage(test_dir, tests, env)

        command = [
            "ctest",
            "--test-dir",
            str(test_dir),
            *args,
            "--output-on-failure",
            "--no-tests=error",
        ]
        print(f"++ Exec {shlex.join(command)}", flush=True)
        executed = subprocess.run(
            command, env=env, check=True, capture_output=True, text=True
        )
        output = executed.stdout + executed.stderr
        print(output, end="")
        _validate_ctest_execution(output, len(names))
        return 0
    except (
        json.JSONDecodeError,
        KeyError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        if isinstance(error, subprocess.CalledProcessError):
            if error.stdout:
                print(error.stdout, end="")
            if error.stderr:
                print(error.stderr, end="", file=sys.stderr)
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
