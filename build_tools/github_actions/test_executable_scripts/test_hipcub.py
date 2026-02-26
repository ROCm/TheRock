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

# If smoke tests are enabled, we run smoke tests only.
# Otherwise, we run the normal test suite
environ_vars = os.environ.copy()
test_type = os.getenv("TEST_TYPE", "full")
if test_type == "smoke":
    environ_vars["GTEST_FILTER"] = ":".join(SMOKE_TESTS)

# Ensure that LD_LIBRARY_PATH contains the path to
# libamdhip64.so, which the resource spec generation (below)
# will use.
THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
rocm_base = Path(THEROCK_BIN_DIR).resolve().parent
ld_path_str = f"{rocm_base}/lib"
existing_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
environ_vars["LD_LIBRARY_PATH"] = (
    f"{ld_path_str}:{existing_ld_path}" if existing_ld_path else ld_path_str
)

# Generate the resource spec file for ctest
resource_spec_file = "resources.json"

res_gen_cmd = [
    f"{THEROCK_BIN_DIR}/hipcub/generate_resource_spec",
    f"{THEROCK_BIN_DIR}/hipcub/{resource_spec_file}",
]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(res_gen_cmd)}")
subprocess.run(res_gen_cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)

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
    "--timeout",
    "300",
]

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)
