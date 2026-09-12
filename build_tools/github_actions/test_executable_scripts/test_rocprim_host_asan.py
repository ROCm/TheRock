#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run only rocPRIM executables proven independent of a GPU device."""

import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROCPRIM_HOST_TESTS = (
    "test_accumulator_t",
    "test_bit_cast",
    "test_invoke_result",
    "test_no_half_operators",
    "test_rocprim_tuple",
    "test_rocprim_types",
    "test_type_traits_interface_cpp17",
    "test_type_traits_interface_cpp20",
    "test_type_traits_interface_gnupp17",
    "test_type_traits_interface_gnupp20",
    "test_radix_key_codec",
)


def main() -> int:
    bin_dir = Path(os.environ["THEROCK_BIN_DIR"]).resolve()
    for binary_name in ROCPRIM_HOST_TESTS:
        executable = bin_dir / binary_name
        if not executable.is_file():
            print(
                f"ERROR: required rocPRIM host-ASAN test is missing: {executable}",
                file=sys.stderr,
            )
            return 1
        command = [str(executable)]
        logging.info("++ Exec %s", shlex.join(command))
        subprocess.run(command, cwd=bin_dir, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
