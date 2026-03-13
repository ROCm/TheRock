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
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

# Import test result collection utilities
sys.path.append(str(THEROCK_DIR / "build_tools" / "github_actions"))
from github_actions_utils import run_test

PLATFORM = os.getenv("PLATFORM")
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")

# GTest sharding
SHARD_INDEX = os.getenv("SHARD_INDEX", 1)
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", 1)
envion_vars = os.environ.copy()
# For display purposes in the GitHub Action UI, the shard array is 1th indexed. However for shard indexes, we convert it to 0th index.
envion_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
envion_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)

logging.basicConfig(level=logging.INFO)

tests_to_exclude = [
    "*known_bug*",
    "*HEEVD*float_complex*",
    "*HEEVJ*float_complex*",
    "*HEGVD*float_complex*",
    "*HEGVJ*float_complex*",
    "*HEEVDX*float_complex*",
    "*SYTRF*float_complex*",
    "*HEEVD*double_complex*",
    "*HEEVJ*double_complex*",
    "*HEGVD*double_complex*",
    "*HEGVJ*double_complex*",
    "*HEEVDX*double_complex*",
    "*SYTRF*double_complex*",
    # TODO(#2824): Re-enable test once flaky issue is resolved
    "checkin_lapack/POTRF_FORTRAN.batched__float_complex/9",
]

exclusion_list = ":".join(tests_to_exclude)

# Create temp file for JSON output
gtest_json_path = Path(tempfile.gettempdir()) / "hipsolver_test_results.json"

cmd = [
    f"{THEROCK_BIN_DIR}/hipsolver-test",
    f"--gtest_output=json:{gtest_json_path}",
    f"--gtest_filter=-{exclusion_list}",
]

run_test(
    cmd,
    output_format="gtest",
    output_path=gtest_json_path,
    cwd=THEROCK_DIR,
    env=envion_vars,
)
