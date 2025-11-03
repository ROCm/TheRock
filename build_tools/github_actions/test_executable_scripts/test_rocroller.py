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

# Runtime libs
if platform == "linux":
    THEROCK_DIST_DIR = BUILD_DIR / "core" / "clr" / "dist"
    llvm_libdir = THEROCK_DIST_DIR / "lib" / "llvm" / "lib"  # libomp.so
    ld_parts = [
        str(THEROCK_DIST_DIR / "lib"),
        str(THEROCK_DIST_DIR / "lib64"),
        str(llvm_libdir),
        # superbuild libs if running from the build tree:
        str(test_bin.parent.parent),  # .../rocRoller/build
        str(BUILD_DIR / "math-libs" / "BLAS" / "rocRoller" / "stage" / "lib"),
        str(BUILD_DIR / "math-libs" / "BLAS" / "rocRoller" / "dist" / "lib"),
    ]
    # De-dupe while preserving order
    seen, ld_clean = set(), []
    for p in ld_parts:
        if p and p not in seen:
            seen.add(p)
            ld_clean.append(p)
    env["ROCM_PATH"] = str(THEROCK_DIST_DIR)
    env["HIP_PATH"] = str(THEROCK_DIST_DIR)

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

cmd = [str(test_bin)]
if test_filter_arg:
    cmd.append(test_filter_arg)

extra = os.getenv("EXTRA_GTEST_ARGS", "")
if extra:
    cmd += shlex.split(extra)

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=str(THEROCK_DIR), check=True, env=env)
