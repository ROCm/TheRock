import logging
import os
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

# TODO#(2836): Re-enable test once issues are resolved
TEST_TO_IGNORE = {
    "gfx1151": {"windows": ["rocprim.device_merge_sort", "rocprim.device_radix_sort"]}
}

SMOKE_TESTS = [
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


cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/rocprim",
    "--output-on-failure",
    "--parallel",
    "8",
    "--timeout",
    "900",
    "--repeat",
    "until-pass:6",
    # shards the tests by running a specific set of tests based on starting test (shard_index) and stride (total_shards)
    "--tests-information",
    f"{shard_index},,{total_shards}",
]

if AMDGPU_FAMILIES in TEST_TO_IGNORE and os_type in TEST_TO_IGNORE[AMDGPU_FAMILIES]:
    ignored_tests = TEST_TO_IGNORE[AMDGPU_FAMILIES][os_type]
    cmd.extend(["--exclude-regex", "|".join(ignored_tests)])

# If smoke tests are enabled, we run smoke tests only.
# Otherwise, we run the normal test suite
environ_vars = os.environ.copy()
test_type = os.getenv("TEST_TYPE", "full")
if test_type == "smoke":
    environ_vars["GTEST_FILTER"] = ":".join(SMOKE_TESTS)

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)
