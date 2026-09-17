#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run the exact device-free HIP Catch2 inventory under host TSAN."""

import hashlib
import logging
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path

from host_tsan_instrumentation import (
    native_host_tsan_environment,
    require_direct_clang_tsan,
    require_no_gpu_nodes,
)


VECTOR_HOST_FILTER = ",".join(
    (
        "Unit_make_vector_SanityCheck_Basic_Host*",
        "Unit_VectorAndVectorOperations_SanityCheck_Basic_Host*",
        "Unit_VectorAndValueTypeOperations_SanityCheck_Basic_Host*",
        "Unit_VectorStructuredBindings_SanityCheck_Basic_host*",
        "Unit_Vector_alignment_check",
        "Unit_Vector_size_check",
        "Unit_dim3_Empty_Positive_Host",
        "Unit_dim3_X_Positive_Host",
        "Unit_dim3_XY_Positive_Host",
        "Unit_dim3_XYZ_Positive_Host",
    )
)

ERROR_NAME_FILTER = ",".join(
    (
        "Unit_hipGetErrorName_Positive_Basic",
        "Unit_hipGetErrorName_Negative_Parameters",
        "Unit_hipGetErrorString_Positive_Basic",
        "Unit_hipGetErrorString_Negative_Parameters",
        "Unit_hipDrvGetErrorName_Positive_Basic",
        "Unit_hipDrvGetErrorName_Negative_Parameters",
        "Unit_hipDrvGetErrorString_Positive_Basic",
        "Unit_hipDrvGetErrorString_Negative_Parameters",
    )
)

HIP_HOST_TESTS = (
    (
        "VectorTypesTest",
        (VECTOR_HOST_FILTER,),
        44,
        "4df54143fe42cba3c7d064daef51f4556bb351066a64f9d1f494fc691302c18b",
    ),
    (
        "ErrorHandlingTest",
        (ERROR_NAME_FILTER,),
        8,
        "36fdea2d27c315d7070de2f86004e8f3485826b9ec547b96b01eeceb64b54d0e",
    ),
    (
        "ChannelDescriptorTest",
        (),
        56,
        "5c0db148b5623da771ac3daaafab0db5a3ce937878b6630ee5b5a0977aefa5f5",
    ),
    (
        "ComplexTest",
        ("*Host*",),
        13,
        "cf08ffdf01ed75347bf52d87f46968e85303db85b21771802899018487f70762",
    ),
)

_LIST_COUNT_RE = re.compile(r"(?m)^\s*(\d+)(?: matching)? test cases?\s*$")
_PASS_COUNT_RE = re.compile(
    r"All tests passed \([^\n]*\bin (\d+) test cases?\)"
)


def _without_aslr(*command: str) -> list[str]:
    """Return a command with ASLR disabled for Clang TSAN's fixed mappings."""
    return ["setarch", platform.machine(), "-R", *command]


def _require_count(output: str, expected: int, pattern: re.Pattern, phase: str) -> None:
    matches = pattern.findall(output)
    if not matches or int(matches[-1]) != expected:
        raise RuntimeError(
            f"HIP host-TSAN {phase} inventory changed: expected {expected}; "
            f"observed {matches[-1] if matches else 'no count'}"
        )
    lowered = output.lower()
    forbidden = ("no tests ran", "no test cases matched", "skipped")
    if any(marker in lowered for marker in forbidden):
        raise RuntimeError(f"HIP host-TSAN {phase} did not execute exactly: {output}")


def _require_inventory(output: str, expected_count: int, expected_digest: str) -> None:
    names = sorted(
        line[2:].rstrip()
        for line in output.splitlines()
        if line.startswith("  ") and len(line) > 2 and not line[2].isspace()
    )
    normalized = "\n".join(names) + "\n"
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    if len(names) != expected_count or digest != expected_digest:
        raise RuntimeError(
            "HIP host-TSAN discovery inventory changed: "
            f"expected {expected_count}/{expected_digest}; got {len(names)}/{digest}"
        )


def main() -> int:
    try:
        require_no_gpu_nodes()
        rocm_root = Path(os.environ["THEROCK_BIN_DIR"]).resolve().parent
        catch_dir = rocm_root / "share" / "hip" / "catch_tests"
        env = native_host_tsan_environment()

        for binary_name, test_spec, expected_count, expected_digest in HIP_HOST_TESTS:
            executable = catch_dir / binary_name
            require_direct_clang_tsan(executable, env)

            list_command = _without_aslr(
                str(executable), *test_spec, "--list-tests"
            )
            listed = subprocess.run(
                list_command,
                cwd=catch_dir,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            _require_count(
                listed.stdout + listed.stderr,
                expected_count,
                _LIST_COUNT_RE,
                f"{binary_name} discovery",
            )
            _require_inventory(listed.stdout, expected_count, expected_digest)

            command = _without_aslr(
                str(executable), *test_spec, "--reporter", "compact"
            )
            logging.info("++ Exec %s", shlex.join(command))
            executed = subprocess.run(
                command,
                cwd=catch_dir,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            output = executed.stdout + executed.stderr
            print(output, end="")
            _require_count(
                output,
                expected_count,
                _PASS_COUNT_RE,
                f"{binary_name} execution",
            )
        return 0
    except (KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
