#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run only rocThrust executables proven independent of a GPU device."""

import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROCTHRUST_HOST_TESTS = (
    "address_stability.hip",
    "alignment.hip",
    "allocator_aware_policies.hip",
    "complex_various.hip",
    "decompose.hip",
    "dependencies_aware_policies.hip",
    "discard_iterator.hip",
    "is_operator_function_object.hip",
    "metaprogramming.hip",
    "min_and_max.hip",
    "mr_disjoint_pool.hip",
    "mr_new.hip",
    "mr_pool.hip",
    "mr_pool_options.hip",
    "preprocessor.hip",
    "tuple_algorithms.hip",
)


def main() -> int:
    bin_dir = Path(os.environ["THEROCK_BIN_DIR"]).resolve()
    for binary_name in ROCTHRUST_HOST_TESTS:
        executable = bin_dir / binary_name
        if not executable.is_file():
            print(
                f"ERROR: required rocThrust host-ASAN test is missing: {executable}",
                file=sys.stderr,
            )
            return 1
        command = [str(executable)]
        logging.info("++ Exec %s", shlex.join(command))
        subprocess.run(command, cwd=bin_dir, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
