#!/usr/bin/env python3
import logging
import os
import shlex
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

# repo + dirs
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR", "")
platform = os.getenv("RUNNER_OS", "linux").lower()

# Sharding
env = os.environ.copy()
env["GTEST_SHARD_INDEX"] = str(int(os.getenv("SHARD_INDEX", "1")) - 1)
env["GTEST_TOTAL_SHARDS"] = str(int(os.getenv("TOTAL_SHARDS", "1")))

# Decide test binary location:
# 1) If CI staged into THEROCK_BIN_DIR, expect "rocroller-tests" there.
# 2) Else use superbuild path.
bin_candidates = []
if THEROCK_BIN_DIR:
    bin_candidates.append(Path(THEROCK_BIN_DIR) / "rocroller-tests")

BUILD_DIR = Path(os.getenv("THEROCK_BUILD_DIR", THEROCK_DIR / "build"))
bin_candidates.append(
    BUILD_DIR
    / "math-libs"
    / "BLAS"
    / "rocRoller"
    / "build"
    / "test"
    / "rocroller-tests"
)

test_bin = next((p for p in bin_candidates if p.is_file()), None)
if not test_bin:
    raise FileNotFoundError(
        f"rocroller-tests not found in: {', '.join(map(str, bin_candidates))}"
    )

# TEST_TYPE → gtest filter
TEST_TYPE = os.getenv("TEST_TYPE", "full").lower()
test_filter_arg = None
if TEST_TYPE == "smoke":
    # keep this subset (TODO: add more tests)
    smoke_tests = [
        "ErrorFixtureDeathTest.*",
        "ArgumentLoaderTest.*",
        "AssemblerTest.*",
        "ControlGraphTest.*",
        "CommandTest.*",
        "ComponentTest.*",
    ]
    test_filter_arg = "--gtest_filter=" + ":".join(smoke_tests)
elif TEST_TYPE == "quick":
    test_filter_arg = "--gtest_filter=*quick*"

# Append to the existing filter or start a negative-only filter
# TODO(#2030): re-enable these tests once compatible with TheRock
# https://github.com/ROCm/TheRock/issues/2030
_excluded = [
    "AssertTest/GPU_AssertTest.GPU_Assert/28",
    "AssertTest/GPU_AssertTest.GPU_UnconditionalAssert/28",
    "AssertTest/GPU_AssertTest.GPU_Assert/29",
    "AssertTest/GPU_AssertTest.GPU_UnconditionalAssert/29",
    "GPU_KernelTests/GPU_KernelTest.GPU_WholeKernel/1",
]
_exclude_str = ":".join(_excluded)
if test_filter_arg:
    test_filter_arg = f"{test_filter_arg}-{_exclude_str}"
else:
    test_filter_arg = f"--gtest_filter=-{_exclude_str}"

cmd = [str(test_bin)]
if test_filter_arg:
    cmd.append(test_filter_arg)

extra = os.getenv("EXTRA_GTEST_ARGS", "")
if extra:
    cmd += shlex.split(extra)

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=str(THEROCK_DIR), check=True, env=env)
