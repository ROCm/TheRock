import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO)

TESTS_TO_IGNORE = "'rocprim.lookback_reproducibility|rocprim.linking|rocprim.device_merge_inplace|rocprim.device_merge_sort|rocprim.device_partition|rocprim.device_radix_sort|rocprim.device_select'"
SMOKE_TESTS = "TestHipGraphBasic:*BasicTests.GetVersion:*ArgIndexIterator:*ExchangeTests*:*HistogramAtomic*:*HistogramSortInput*:*VectorizationTests*:*FirstPart:*ceIntegral/*:*tyIntegral/*:*SecondPart/*:*ThirdPart/*:*MergeTests/*:*RadixSortIntegral/*:*ReduceSingleValueTestsIntegral:*ReduceInputArrayTestsIntegral/*:*ReduceSingleValueTestsFloating:*ReduceInputArrayTestsFloating:*HipcubBlockRunLengthDecodeTest/*:*BlockScan:*ShuffleTestsIntegral*:*ShuffleTestsFloating/*:*SortBitonicTestsIntegral/*:*ConfigDispatchTests.*:*ConstantIteratorTests/*:*CountingIteratorTests/*:*BatchMemcpyTests/*:*Histogram*:*PartitionTests/*:*PartitionLargeInputTest/*:*ReduceByKey*:*ReduceTests/*:*ReducePrecisionTests/*:*RunLengthEncode/*:*DeviceScanTests/*:*SegmentedReduce/*:*SelectTests/*:*SelectLargeInputFlaggedTest/*:*TransformTests/*:*DiscardIteratorTests.Less:*RadixKeyCodecTest.*:*RadixMergeCompareTest/*:*RadixSort/*:*PredicateIteratorTests.*:*ReverseIteratorTests.*:*ThreadTests/*:*ThreadOperationTests/*:*TransformIteratorTests/*:*IntrinsicsTests*:*InvokeResultBinOpTests/*:*InvokeResultUnOpTests/*:*WarpExchangeTest/*:*WarpExchangeScatterTest/*:*WarpLoadTest/*:*WarpReduceTestsIntegral/*:*WarpReduceTestsFloating/*:*WarpScanTests*:*WarpSortShuffleBasedTestsIntegral/*"

cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/rocprim",
    "--output-on-failure",
    "--parallel",
    "8",
    "--exclude-regex",
    TESTS_TO_IGNORE,
    "--timeout",
    "900",
    "--repeat",
    "until-pass:6",
]

# If smoke tests are enabled, we run smoke tests only.
# Otherwise, we run the normal test suite
environ_vars = os.environ.copy()
test_type = os.getenv("TEST_TYPE", "all")
if test_type == "smoke":
    environ_vars["GTEST_FILTER"] = f"'{SMOKE_TESTS}'"

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)
