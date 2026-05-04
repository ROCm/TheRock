# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
import platform
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO)

QUICK_TESTS = [
    "*ShuffleTests/*.*",
    "*WarpStoreTest/*.*",
    "AdjacentDifference/*.*",
    "AdjacentDifferenceSubtract/*.*",
    "BatchCopyTests/*.*",
    "BatchMemcpyTests/*.*",
    "BlockScan*",
    "DeviceScanTests/*.*",
    "Discontinuity/*.*",
    "DivisionOperatorTests/*.*",
    "ExchangeTests",
    "GridTests/*.*",
    "HistogramEven/*.*",
    "HistogramInputArrayTests/*.*",
    "HistogramRange/*.*",
    "IteratorTests/*.*",
    "LoadStoreTestsDirect/*.*",
    "LoadStoreTestsStriped/*.*",
    "LoadStoreTestsTranspose/*.*",
    "LoadStoreTestsVectorize/*.*",
    "MergeSort/*.*",
    "NCThreadOperatorsTests/*",
    "RadixRank/*.*",
    "RadixSort/*.*",
    "ReduceArgMinMaxSpecialTests/*.*",
    "ReduceInputArrayTests/*.*",
    "ReduceLargeIndicesTests/*.*",
    "ReduceSingleValueTests/*.*",
    "ReduceTests/*.*",
    "RunLengthDecodeTest/*.*",
    "RunLengthEncode/*.*",
    "SegmentedReduce/*.*",
    "SegmentedReduceArgMinMaxSpecialTests/*.*",
    "SegmentedReduceOp/*.*",
    "SelectTests/*.*",
    "ThreadOperationTests/*.*",
    "ThreadOperatorsTests/*.*",
    "UtilPtxTests/*.*",
    "WarpExchangeTest/*.*",
    "WarpLoadTest/*.*",
    "WarpMergeSort/*.*",
    "WarpReduceTests/*.*",
    "WarpScanTests*",
]

# Generate the resource spec file for ctest
rocm_base = Path(THEROCK_BIN_DIR).resolve().parent
ld_paths = [
    rocm_base / "lib",
]
ld_paths_str = os.pathsep.join(str(p) for p in ld_paths)
existing_path = os.environ.get("PATH", "")
existing_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
env_vars = os.environ.copy()
env_vars["PATH"] = (
    f"{THEROCK_BIN_DIR}{os.pathsep}{existing_path}"
    if existing_path
    else THEROCK_BIN_DIR
)
env_vars["ROCM_PATH"] = str(rocm_base)
env_vars["LD_LIBRARY_PATH"] = (
    f"{ld_paths_str}{os.pathsep}{existing_ld_path}"
    if existing_ld_path
    else ld_paths_str
)

is_windows = platform.system() == "Windows"
exe_name = "generate_resource_spec.exe" if is_windows else "generate_resource_spec"
exe_dir = rocm_base / "bin" / "hipcub"
resource_spec_file = "resources.json"

res_gen_cmd = [
    str(exe_dir / exe_name),
    str(exe_dir / resource_spec_file),
]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(res_gen_cmd)}")
subprocess.run(res_gen_cmd, cwd=THEROCK_DIR, check=True, env=env_vars)

# Run ctest with resource spec file
cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/hipcub",
    "--output-on-failure",
    "--parallel",
    "8",
    "--resource-spec-file",
    resource_spec_file,
]

# If quick tests are enabled, we run quick tests only.
# Otherwise, we run the normal test suite
environ_vars = os.environ.copy()
test_type = os.getenv("TEST_TYPE", "full")
if test_type == "quick":
    environ_vars["GTEST_FILTER"] = ":".join(QUICK_TESTS)

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)
