import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO)

SMOKE_TESTS = [
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

cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/hipcub",
    "--output-on-failure",
    "--parallel",
    "8",
    "--timeout",
    "300",
    "--repeat",
    "until-pass:3",
]

# If smoke tests are enabled, we run smoke tests only.
# Otherwise, we run the normal test suite
environ_vars = os.environ.copy()
test_type = os.getenv("TEST_TYPE", "full")
if test_type == "smoke":
    environ_vars["GTEST_FILTER"] = ":".join(SMOKE_TESTS)

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)
