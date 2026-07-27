# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Thin wrapper that runs MIOpen tests through the standardized test runner.

MIOpen's test selection is owned by the test filter standardization in
rocm-libraries (``projects/miopen/test/gtest/test_categories.yaml``), which is
compiled into an installed ``bin/MIOpen/CTestTestfile.cmake`` and consumed by
the generic ``test_runner.py`` via ctest category labels (quick / standard /
comprehensive / full) plus GPU-specific ``ex_gpu_<arch>`` exclusions.

This convenience script simply delegates to ``test_runner.py`` so the YAML is
the single source of truth.

The test category can be selected with ``--test-type`` (alias ``--category``)
or the ``TEST_TYPE`` environment variable. Precedence is: CLI option, then
``TEST_TYPE``, then the default of "standard". All other configuration
(THEROCK_BIN_DIR, AMDGPU_FAMILIES, SHARD_INDEX, TOTAL_SHARDS, ...) is read by
test_runner.py from the environment.
"""

import argparse
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_RUNNER = SCRIPT_DIR / "test_runner.py"

# Categories defined by the MIOpen test filter standardization.
TEST_CATEGORIES = ["quick", "standard", "comprehensive", "full"]

# Default category when neither --test-type nor TEST_TYPE is provided.
# (GitHub Actions passes an empty string when a workflow input is left blank.)
DEFAULT_TEST_TYPE = "standard"

logging.basicConfig(level=logging.INFO)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run MIOpen tests via the standardized ctest test runner."
    )
    parser.add_argument(
        "--test-type",
        "--category",
        dest="test_type",
        choices=TEST_CATEGORIES,
        default=None,
        help=(
            "Test category to run. Overrides the TEST_TYPE environment "
            f"variable. Defaults to TEST_TYPE or '{DEFAULT_TEST_TYPE}'."
        ),
    )
    args = parser.parse_args()

    # Precedence: CLI option > TEST_TYPE env var > default.
    category = args.test_type or os.getenv("TEST_TYPE") or DEFAULT_TEST_TYPE

    env = os.environ.copy()
    # This script always tests MIOpen; test_runner.py maps "miopen" -> "MIOpen".
    env["TEST_COMPONENT"] = "miopen"
    env["TEST_TYPE"] = category

    logging.info(f"MIOpen test category (TEST_TYPE): {category}")

    cmd = [sys.executable, str(TEST_RUNNER)]
    logging.info(f"++ Exec $ {shlex.join(cmd)}")
    return subprocess.run(cmd, env=env, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
