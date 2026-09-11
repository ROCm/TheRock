#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run only the rocRAND/hipRAND tests proven safe on a CPU-only host."""

import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROCRAND_TESTS = (
    ("test_cpp_utils", ()),
    ("test_log_normal_distribution", ()),
    ("test_normal_distribution", ()),
    ("test_poisson_distribution", ()),
    ("test_rocrand_mrg31k3p_prng", ()),
    ("test_rocrand_mrg32k3a_prng", ()),
    ("test_rocrand_mt19937_octo_engine_prng", ()),
    ("test_rocrand_linkage", ()),
    (
        "test_rocrand_generator_type",
        (
            "--gtest_filter=rocrand_generator_type_tests.rocrand_generator:"
            "rocrand_generator_type_tests.generate_test",
        ),
    ),
)
HIPRAND_TESTS = (("test_hiprand_linkage", ()),)


def main() -> int:
    component = os.environ.get("TEST_COMPONENT", "")
    selections = {"rocrand": ROCRAND_TESTS, "hiprand": HIPRAND_TESTS}
    if component not in selections:
        print(
            f"ERROR: TEST_COMPONENT must be rocrand or hiprand, got {component!r}",
            file=sys.stderr,
        )
        return 1

    bin_dir = Path(os.environ["THEROCK_BIN_DIR"]).resolve()
    for binary_name, arguments in selections[component]:
        executable = bin_dir / binary_name
        if not executable.is_file():
            print(f"ERROR: required host-ASAN test is missing: {executable}", file=sys.stderr)
            return 1
        command = [str(executable), *arguments]
        logging.info("++ Exec %s", shlex.join(command))
        subprocess.run(command, cwd=bin_dir, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
