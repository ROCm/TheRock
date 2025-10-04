import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO)

SMOKE_TESTS = "AdjacentDifference/*.*:AdjacentDifferenceSubtract/*.*:Discontinuity/*.*:ExchangeTests:HistogramInputArrayTests/*.*:LoadStoreTestsDirect/*.*:LoadStoreTestsVectorize/*.*:LoadStoreTestsTranspose/*.*:LoadStoreTestsStriped/*.*:MergeSort/*.*:RadixRank/*.*:RadixSort/*.*:ReduceSingleValueTests/*.*:ReduceInputArrayTests/*.*:RunLengthDecodeTest/*.*:BlockScan*:*ShuffleTests/*.*:BatchCopyTests/*.*:HistogramEven/*.*:HistogramRange/*.*:BatchMemcpyTests/*.*:ReduceTests/*.*:ReduceArgMinMaxSpecialTests/*.*:ReduceLargeIndicesTests/*.*:RunLengthEncode/*.*:DeviceScanTests/*.*:SegmentedReduce/*.*:SegmentedReduceOp/*.*:SegmentedReduceArgMinMaxSpecialTests/*.*:SelectTests/*.*:GridTests/*.*:UtilPtxTests/*.*:WarpExchangeTest/*.*:WarpLoadTest/*.*:WarpMergeSort/*.*:WarpReduceTests/*.*:WarpScanTests*:*WarpStoreTest/*.*:IteratorTests/*.*:ThreadOperationTests/*.*:ThreadOperatorsTests/*.*:DivisionOperatorTests/*.*:NCThreadOperatorsTests/*"

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
test_type = os.getenv("TEST_TYPE", "all")
if test_type == "smoke":
    environ_vars["GTEST_FILTER"] = f"'{SMOKE_TESTS}'"

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)
