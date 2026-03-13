# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")
platform = os.getenv("RUNNER_OS").lower()
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

# Import test result collection utilities
sys.path.append(str(THEROCK_DIR / "build_tools" / "github_actions"))
from github_actions_utils import run_test

# GTest sharding
SHARD_INDEX = os.getenv("SHARD_INDEX", 1)
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", 1)
environ_vars = os.environ.copy()
# For display purposes in the GitHub Action UI, the shard array is 1th indexed. However for shard indexes, we convert it to 0th index.
environ_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
environ_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)

# Enable GTest "brief" output: only show failures and the final results
environ_vars["GTEST_BRIEF"] = str(1)

# Some of our runtime kernel compilations have been relying on either ROCM_PATH being set, or ROCm being installed at
# /opt/rocm. Neither of these is true in TheRock so we need to supply ROCM_PATH to our tests.
ROCM_PATH = Path(THEROCK_BIN_DIR).resolve().parent
environ_vars["ROCM_PATH"] = str(ROCM_PATH)

logging.basicConfig(level=logging.INFO)

# If smoke tests are enabled, we run smoke tests only.
# Otherwise, we run the normal test suite
test_type = os.getenv("TEST_TYPE", "full")

# TODO(#2823): Re-enable test once flaky issue is resolved
TESTS_TO_IGNORE = ["unpack_util_test"]

test_subdir = ""
timeout = "3600"
if test_type == "smoke":
    # The emulator regression tests are very fast.
    # If we need something even faster we can use "/smoke" here.
    test_subdir = "/regression"
    timeout = "720"
elif test_type == "regression":
    test_subdir = "/regression"
    timeout = "720"

# Make per-device adjustments
ctest_parallelism = "2"
if AMDGPU_FAMILIES == "gfx1153":
    ctest_parallelism = "1"

# Create temp file for JUnit XML output
junit_xml_path = Path(tempfile.gettempdir()) / "rocwmma_test_results.xml"

cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/rocwmma{test_subdir}",
    "--output-on-failure",
    "--output-junit",
    str(junit_xml_path),
    "--parallel",
    ctest_parallelism,
    "--timeout",
    timeout,
    "--exclude-regex",
    "|".join(TESTS_TO_IGNORE),
]
run_test(
    cmd,
    output_format="ctest",
    output_path=junit_xml_path,
    cwd=THEROCK_DIR,
    env=environ_vars,
)
