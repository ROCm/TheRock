#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build and run RPP's exact CPU-only test inventory under host TSAN."""

import hashlib
import json
import os
import platform
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


EXPECTED_NAMES = (
    "rpp_sanity_test_brightness_host_f32",
    "rpp_qa_tests_tensor_image_host_all",
    "rpp_qa_tests_tensor_misc_host_all",
)
EXECUTED_NAMES = EXPECTED_NAMES[1:]
EXPECTED_INVENTORY_SHA256 = (
    "0e0c49f9da7bd690d8853f9347d454b010b743ee8f96a353afffe5365ec9a521"
)
TEST_REGEX = "^(" + "|".join(EXPECTED_NAMES) + ")$"
_CTEST_NAME_RE = re.compile(r"^\s*Test\s+#\d+:\s+(.+?)\s*$")
_RPP_CHILD_FAILURE_RE = re.compile(
    r"Returned non-zero exit status\s*:|"
    r"(?:WARNING|FATAL|SUMMARY): ThreadSanitizer|"
    r"Segmentation fault",
    re.IGNORECASE,
)


def _compiler(prefix: Path, name: str) -> Path:
    candidates = (
        prefix / "llvm" / "bin" / name,
        prefix / "lib" / "llvm" / "bin" / name,
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError(f"artifact compiler not found: {name}")


def _test_environment(prefix: Path) -> dict[str, str]:
    env = native_host_tsan_environment()
    env["ROCM_PATH"] = str(prefix)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [
            str(prefix / "lib"),
            str(prefix / "lib" / "rocm_sysdeps" / "lib"),
            str(prefix / "lib" / "llvm" / "lib"),
            env.get("LD_LIBRARY_PATH", ""),
        ]
    ).rstrip(os.pathsep)
    # These comprehensive host suites require concurrent OpenMP workers to
    # finish in a practical amount of time. Container affinity can expose only
    # one CPU even when the pod may run multiple workers, so do not derive this
    # value from sched_getaffinity or inherited host settings.
    env["OMP_NUM_THREADS"] = "4"
    env["OPENBLAS_NUM_THREADS"] = "1"
    return env


def _run(
    command: list[str], env: dict[str, str], cwd: Path, capture: bool = False
) -> subprocess.CompletedProcess:
    print(f"++ Exec [{cwd}]$ {shlex.join(command)}", flush=True)
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        capture_output=capture,
        text=True,
    )


def _parse_names(output: str) -> list[str]:
    return [
        match.group(1)
        for line in output.splitlines()
        if (match := _CTEST_NAME_RE.match(line))
    ]


def _validate_inventory(names: list[str]) -> None:
    normalized = "".join(f"{name}\n" for name in sorted(names))
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    if (
        tuple(sorted(names)) != tuple(sorted(EXPECTED_NAMES))
        or digest != EXPECTED_INVENTORY_SHA256
    ):
        raise RuntimeError(
            "RPP host-TSAN inventory changed: "
            f"expected count=3, sha256={EXPECTED_INVENTORY_SHA256}; "
            f"got count={len(names)}, sha256={digest}, names={sorted(names)}"
        )


def _selected_commands(build_dir: Path, env: dict[str, str]) -> list[dict]:
    result = _run(
        [
            "ctest",
            "--test-dir",
            str(build_dir),
            "--show-only=json-v1",
            "-R",
            TEST_REGEX,
        ],
        env,
        build_dir,
        capture=True,
    )
    tests = json.loads(result.stdout).get("tests", [])
    if sorted(test.get("name") for test in tests) != sorted(EXPECTED_NAMES):
        raise RuntimeError("RPP CTest JSON inventory changed")
    return tests


def _run_ctest(
    build_dir: Path,
    env: dict[str, str],
    name: str,
    *,
    timeout_seconds: int,
) -> None:
    command = [
        "setarch",
        platform.machine(),
        "-R",
        "ctest",
        "--test-dir",
        str(build_dir),
        "-R",
        f"^{re.escape(name)}$",
        "--output-on-failure",
        "--no-tests=error",
        "--timeout",
        str(timeout_seconds),
    ]
    executed = _run(command, env, build_dir, capture=True)
    output = executed.stdout + executed.stderr
    print(output, end="")
    if re.search(r"\*\*\*Skipped|Not Run|did not run", output, re.IGNORECASE):
        raise RuntimeError(f"RPP host-TSAN execution skipped {name}")
    last_test_log = _read_last_test_log(build_dir)
    if _RPP_CHILD_FAILURE_RE.search(output + "\n" + last_test_log):
        raise RuntimeError(f"RPP host-TSAN child process failed: {name}")
    if not re.search(r"100% tests passed,\s+0 tests failed out of 1\b", output):
        raise RuntimeError(f"RPP host-TSAN test did not pass: {name}")


def _read_last_test_log(build_dir: Path) -> str:
    path = build_dir / "Testing" / "Temporary" / "LastTest.log"
    if not path.is_file():
        raise RuntimeError(f"RPP CTest did not produce its execution log: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def _brightness_payload(tests: list[dict]) -> list[str]:
    test = next(
        (test for test in tests if test.get("name") == EXPECTED_NAMES[0]),
        None,
    )
    if test is None or not isinstance(test.get("command"), list):
        raise RuntimeError("RPP brightness CTest command is missing")
    command = test["command"]
    try:
        marker = command.index("--test-command")
    except ValueError as error:
        raise RuntimeError("RPP brightness CTest wrapper changed") from error
    payload = command[marker + 1 :]
    if not payload or Path(payload[0]).name != "Tensor_image_host":
        raise RuntimeError("RPP brightness CTest payload changed")
    return payload[1:]


def _run_brightness(
    build_dir: Path, env: dict[str, str], tests: list[dict]
) -> None:
    _run(
        [
            "setarch",
            platform.machine(),
            "-R",
            str(build_dir / "HOST" / "Tensor_image_host"),
            *_brightness_payload(tests),
        ],
        env,
        build_dir,
    )


def main() -> int:
    try:
        require_no_gpu_nodes()
        prefix = Path(os.environ["THEROCK_BIN_DIR"]).resolve().parent
        source_dir = prefix / "share" / "rpp" / "test"
        if not source_dir.is_dir():
            raise RuntimeError(f"installed RPP test source is missing: {source_dir}")

        env = _test_environment(prefix)

        c_compiler = _compiler(prefix, "amdclang")
        cxx_compiler = _compiler(prefix, "amdclang++")
        compile_flags = "-fsanitize=thread -fno-omit-frame-pointer"
        link_flags = "-fsanitize=thread -shared-libsan"

        with tempfile.TemporaryDirectory(prefix="rpp-host-tsan-") as temp:
            build_dir = Path(temp)
            _run(
                [
                    "cmake",
                    "-GNinja",
                    f"-DROCM_PATH={prefix}",
                    "-DUSING_THE_ROCK=ON",
                    "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
                    f"-DCMAKE_C_COMPILER={c_compiler}",
                    f"-DCMAKE_CXX_COMPILER={cxx_compiler}",
                    f"-DCMAKE_C_FLAGS={compile_flags}",
                    f"-DCMAKE_CXX_FLAGS={compile_flags}",
                    f"-DCMAKE_EXE_LINKER_FLAGS={link_flags}",
                    str(source_dir),
                ],
                env,
                build_dir,
            )
            _run(
                [
                    "cmake",
                    "--build",
                    str(build_dir),
                    "--target",
                    "Tensor_image_host",
                    "Tensor_misc_host",
                ],
                env,
                build_dir,
            )

            for binary in (
                build_dir / "HOST" / "Tensor_image_host",
                build_dir / "HOST" / "Tensor_misc_host",
            ):
                require_direct_clang_tsan(binary, env)
            listed = _run(
                ["ctest", "--test-dir", str(build_dir), "-N", "-R", TEST_REGEX],
                env,
                build_dir,
                capture=True,
            )
            names = _parse_names(listed.stdout)
            _validate_inventory(names)
            selected_tests = _selected_commands(build_dir, env)

            # The registered brightness smoke uses CTest --build-and-test to
            # reconfigure the already configured HOST build directory in place;
            # that wrapper deterministically corrupts/crashes on the shared AWS
            # runner. Execute the already-built, linkage-verified binary with
            # the exact registered brightness F32 arguments instead.
            _run_brightness(build_dir, env, selected_tests)

            # The comprehensive image and misc drivers are intentionally
            # serial: both recreate the same nested build directory. The cloud
            # CPU runner needs more than CTest's previous 600-second allowance.
            # Their child-process logs are scanned explicitly because the
            # upstream drivers currently print child failures without exiting
            # nonzero themselves.
            for name in EXECUTED_NAMES:
                _run_ctest(build_dir, env, name, timeout_seconds=1500)
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
