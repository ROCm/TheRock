# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import platform
import shlex
import subprocess
from pathlib import Path
import platform

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")
os_type = platform.system().lower()

logging.basicConfig(level=logging.INFO)

TEST_TO_IGNORE = {
    # TODO(#2836): Re-enable gfx110X tests once issues are resolved
    "gfx110X-all": {
        "windows": [
            "rocprim.block_discontinuity",
            "rocprim.device_merge_sort",
            "rocprim.device_reduce",
        ]
    },
    "gfx1151": {
        "windows": [
            # TODO(#2836): Re-enable test once issues are resolved
            "rocprim.device_merge_sort",
            # TODO(#2836): Re-enable test once issues are resolved
            "rocprim.device_radix_sort",
        ]
    },
}

QUICK_TESTS = [
    "*ArgIndexIterator",
    "*BasicTests.GetVersion",
    "*BatchMemcpyTests/*",
    "*BlockScan",
    "*ConfigDispatchTests.*",
    "*ConstantIteratorTests/*",
    "*CountingIteratorTests/*",
    "*DeviceScanTests/*",
    "*DiscardIteratorTests.Less",
    "*ExchangeTests*",
    "*FirstPart",
    "*HipcubBlockRunLengthDecodeTest/*",
    "*Histogram*",
    "*HistogramAtomic*",
    "*HistogramSortInput*",
    "*IntrinsicsTests*",
    "*InvokeResultBinOpTests/*",
    "*InvokeResultUnOpTests/*",
    "*MergeTests/*",
    "*PartitionLargeInputTest/*",
    "*PartitionTests/*",
    "*PredicateIteratorTests.*",
    "*RadixKeyCodecTest.*",
    "*RadixMergeCompareTest/*",
    "*RadixSort/*",
    "*RadixSortIntegral/*",
    "*ReduceByKey*",
    "*ReduceInputArrayTestsFloating",
    "*ReduceInputArrayTestsIntegral/*",
    "*ReducePrecisionTests/*",
    "*ReduceSingleValueTestsFloating",
    "*ReduceSingleValueTestsIntegral",
    "*ReduceTests/*",
    "*ReverseIteratorTests.*",
    "*RunLengthEncode/*",
    "*SecondPart/*",
    "*SegmentedReduce/*",
    "*SelectLargeInputFlaggedTest/*",
    "*SelectTests/*",
    "*ShuffleTestsFloating/*",
    "*ShuffleTestsIntegral*",
    "*SortBitonicTestsIntegral/*",
    "*ThirdPart/*",
    "*ThreadOperationTests/*",
    "*ThreadTests/*",
    "*TransformIteratorTests/*",
    "*TransformTests/*",
    "*VectorizationTests*",
    "*WarpExchangeScatterTest/*",
    "*WarpExchangeTest/*",
    "*WarpLoadTest/*",
    "*WarpReduceTestsFloating/*",
    "*WarpReduceTestsIntegral/*",
    "*WarpScanTests*",
    "*WarpSortShuffleBasedTestsIntegral/*",
    "*ceIntegral/*",
    "*tyIntegral/*",
    "TestHipGraphBasic",
]

# sharding
shard_index = int(os.getenv("SHARD_INDEX", "1")) - 1
total_shards = int(os.getenv("TOTAL_SHARDS", "1"))

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
exe_dir = rocm_base / "bin" / "rocprim"

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
    f"{THEROCK_BIN_DIR}/rocprim",
    "--output-on-failure",
    "--resource-spec-file",
    resource_spec_file,
    # shards the tests by running a specific set of tests based on starting test (shard_index) and stride (total_shards)
    "--tests-information",
    f"{shard_index},,{total_shards}",
]

if AMDGPU_FAMILIES in TEST_TO_IGNORE and os_type in TEST_TO_IGNORE[AMDGPU_FAMILIES]:
    ignored_tests = TEST_TO_IGNORE[AMDGPU_FAMILIES][os_type]
    cmd.extend(["--exclude-regex", "|".join(ignored_tests)])

# If quick tests are enabled, we run quick tests only.
# Otherwise, we run the standard test suite.
environ_vars = os.environ.copy()
test_type = os.getenv("TEST_TYPE", "standard")
if test_type == "quick":
    environ_vars["GTEST_FILTER"] = ":".join(QUICK_TESTS)

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)
