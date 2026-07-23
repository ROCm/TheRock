#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""UCCL ROCm Smoke Tests Runner.

Runs lightweight smoke tests to verify that the UCCL wheel is installed
correctly, GPU hardware is accessible, and the basic UCCL Python API
is importable.

Usage Examples
--------------
Basic usage (auto-detect GPU):
    $ python run_uccl_smoke_tests.py

Specify GPU family:
    $ python run_uccl_smoke_tests.py --amdgpu-family gfx942

Pass additional pytest arguments after "--":
    $ python run_uccl_smoke_tests.py -- --tb=short -x
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

THIS_SCRIPT_DIR = Path(__file__).resolve().parent


def cmd_arguments(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    try:
        rest_pos = argv.index("--")
    except ValueError:
        passthrough_pytest_args = []
    else:
        passthrough_pytest_args = argv[rest_pos + 1 :]
        argv = argv[:rest_pos]

    parser = argparse.ArgumentParser(
        description="Runs UCCL smoke-tests for AMD GPUs. "
        'All arguments after "--" are passed directly to pytest.'
    )

    parser.add_argument(
        "--amdgpu-family",
        type=str,
        default=os.getenv("AMDGPU_FAMILY", ""),
        help='AMDGPU family (e.g. "gfx942"). Used to select GPU via '
        "HIP_VISIBLE_DEVICES before running tests.",
    )

    args = parser.parse_args(argv)
    return args, passthrough_pytest_args


def main() -> int:
    args, passthrough_pytest_args = cmd_arguments(sys.argv[1:])

    smoke_tests_dir = THIS_SCRIPT_DIR / "smoke-tests"
    if not smoke_tests_dir.exists():
        print(f"ERROR: Smoke test directory '{smoke_tests_dir}' does not exist.")
        return 1

    # Build pytest command. We invoke pytest as a subprocess rather than
    # via pytest.main() so that HIP_VISIBLE_DEVICES (if set externally)
    # takes effect before torch is imported.
    pytest_cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(smoke_tests_dir),
    ]
    pytest_cmd.extend(passthrough_pytest_args)

    print(f"Running UCCL smoke tests from {smoke_tests_dir}")
    print(f"Command: {' '.join(pytest_cmd)}")

    result = subprocess.run(pytest_cmd)
    print(f"Smoke tests finished with return code: {result.returncode}")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
