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
PRESUBMIT_CASES = {
    # One positive host-only case from every image augmentation family not
    # already covered by the direct brightness/color canary. Each case still
    # exercises U8/F32 and every supported host layout conversion.
    EXPECTED_NAMES[1]: (
        "5",  # pixelate
        "21",  # resize
        "40",  # erode
        "49",  # box_filter
        "61",  # magnitude
        "65",  # bitwise_and
        "70",  # copy
        "90",  # tensor_mean
    ),
    # Keep positive tensor-shape, normalization, logarithmic, composition,
    # logical, addition, and division seams from the smaller misc suite.
    EXPECTED_NAMES[2]: (
        "0",  # transpose
        "1",  # normalize
        "2",  # log
        "3",  # concat
        "5",  # tensor_and_tensor
        "8",  # tensor_add_tensor
        "11",  # tensor_divide_tensor
    ),
}
EXECUTED_NAMES = tuple(PRESUBMIT_CASES)
PRESUBMIT_EXTRA_ARGS = {
    EXPECTED_NAMES[1]: (),
    # Preserve both a low and high tensor rank without running every rank.
    EXPECTED_NAMES[2]: ("--num_dims_list", "2", "4"),
}
PRESUBMIT_EXPECTED_QA = {
    EXPECTED_NAMES[1]: 181,
    EXPECTED_NAMES[2]: 90,
}
PRESUBMIT_TIMEOUT_SECONDS = {
    EXPECTED_NAMES[1]: 900,
    EXPECTED_NAMES[2]: 600,
}
EXPECTED_INVENTORY_SHA256 = (
    "0e0c49f9da7bd690d8853f9347d454b010b743ee8f96a353afffe5365ec9a521"
)
TEST_REGEX = "^(" + "|".join(EXPECTED_NAMES) + ")$"
_CTEST_NAME_RE = re.compile(r"^\s*Test\s+#\d+:\s+(.+?)\s*$")
_RPP_CHILD_FAILURE_RE = re.compile(
    r"Returned non-zero exit status\s*:|"
    r"(?:WARNING|FATAL|SUMMARY): ThreadSanitizer|"
    r"Segmentation fault|Invalid case name or number|"
    r"Traceback \(most recent call last\)|ERROR: QA failures|"
    r"\b(?:SIGSEGV|SIGABRT)\b",
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
    # Keep libomp workers active. Sleeping workers exercise libomp's internal
    # futex/suspension implementation, which is not TSAN-instrumented and
    # produces a false positive in __kmp_lock_suspend_mx.
    env["OMP_WAIT_POLICY"] = "ACTIVE"
    env["KMP_BLOCKTIME"] = "infinite"
    return env


def _run(
    command: list[str],
    env: dict[str, str],
    cwd: Path,
    capture: bool = False,
    timeout_seconds: int | None = None,
) -> subprocess.CompletedProcess:
    print(f"++ Exec [{cwd}]$ {shlex.join(command)}", flush=True)
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        capture_output=capture,
        text=True,
        timeout=timeout_seconds,
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


def _registered_command(tests: list[dict], name: str) -> list[str]:
    test = next((test for test in tests if test.get("name") == name), None)
    if test is None or not isinstance(test.get("command"), list):
        raise RuntimeError(f"RPP registered command is missing: {name}")
    command = test["command"]
    expected_script = {
        EXPECTED_NAMES[1]: "runImageTests.py",
        EXPECTED_NAMES[2]: "runMiscTests.py",
    }[name]
    if len(command) < 2 or Path(command[1]).name != expected_script:
        raise RuntimeError(f"RPP registered command changed: {name}")
    if any(
        argument in command
        for argument in ("--case_list", "--case_start", "--case_end")
    ):
        raise RuntimeError(f"RPP registered command already filters cases: {name}")
    return command


def _run_presubmit_suite(
    build_dir: Path,
    env: dict[str, str],
    tests: list[dict],
    name: str,
    *,
    timeout_seconds: int,
) -> None:
    cases = PRESUBMIT_CASES[name]
    registered = _registered_command(tests, name)
    command = [
        "setarch",
        platform.machine(),
        "-R",
        registered[0],
        "-u",
        *registered[1:],
        *PRESUBMIT_EXTRA_ARGS[name],
        "--case_list",
        *cases,
    ]
    executed = _run(
        command,
        env,
        build_dir,
        capture=True,
        timeout_seconds=timeout_seconds,
    )
    output = executed.stdout + executed.stderr
    print(output, end="")
    if _RPP_CHILD_FAILURE_RE.search(output):
        raise RuntimeError(f"RPP host-TSAN child process failed: {name}")
    requested = re.search(
        r"Total test cases including all subvariants REQUESTED =\s*(\d+)", output
    )
    passed = re.search(
        r"Total test cases including all subvariants PASSED =\s*(\d+)", output
    )
    expected_qa = PRESUBMIT_EXPECTED_QA[name]
    if (
        requested is None
        or passed is None
        or int(requested.group(1)) != expected_qa
        or requested.group(1) != passed.group(1)
    ):
        raise RuntimeError(
            f"RPP host-TSAN QA summary did not pass: {name}; "
            f"expected {expected_qa} requested and passed"
        )

    binary_name = {
        EXPECTED_NAMES[1]: "Tensor_image_host",
        EXPECTED_NAMES[2]: "Tensor_misc_host",
    }[name]
    require_direct_clang_tsan(build_dir / "build" / binary_name, env)


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
        # The registered Python drivers create their own nested CMake build.
        # Carry the same compiler and TSAN flags into that build rather than
        # validating one binary and executing a different, unsanitized one.
        env["CC"] = str(c_compiler)
        env["CXX"] = str(cxx_compiler)
        env["CFLAGS"] = compile_flags
        env["CXXFLAGS"] = compile_flags
        env["LDFLAGS"] = link_flags

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

            # The comprehensive drivers are intentionally serial because both
            # recreate the same nested build directory. Full image QA takes
            # hours under TSAN on the shared cloud CPU, so use an explicit
            # positive presubmit allowlist covering every image family plus the
            # core misc seams. Device cases remain absent. Output is scanned
            # fail-closed because the drivers can print child failures and
            # still exit zero.
            for name in EXECUTED_NAMES:
                _run_presubmit_suite(
                    build_dir,
                    env,
                    selected_tests,
                    name,
                    timeout_seconds=PRESUBMIT_TIMEOUT_SECONDS[name],
                )
        return 0
    except (
        json.JSONDecodeError,
        KeyError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        if isinstance(
            error, (subprocess.CalledProcessError, subprocess.TimeoutExpired)
        ):
            if error.stdout:
                print(error.stdout, end="")
            if error.stderr:
                print(error.stderr, end="", file=sys.stderr)
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
