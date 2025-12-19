import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

logging.basicConfig(level=logging.INFO)

SMOKE_TESTS = [
    "AllocatorTests.*",
    "AsyncExclusiveScan*",
    "AsyncInclusiveScan*",
    "AsyncReduce*",
    "AsyncSort*",
    "AsyncTransform*",
    "AsyncTriviallyRelocatableElements*",
    "ConstantIteratorTests.*",
    "Copy*",
    "CopyN*",
    "Count*",
    "CountingIteratorTests.*",
    "Dereference*",
    "DeviceDelete*",
    "DevicePathSimpleTest",
    "DevicePtrTests.*",
    "DeviceReferenceTests.*",
    "DiscardIteratorTests.*",
    "EqualTests.*",
    "Fill*",
    "Find*",
    "ForEach*",
    "Gather*",
    "Generate*",
    "InnerProduct*",
    "IsPartitioned*",
    "IsSorted*",
    "IsSortedUntil*",
    "MemoryTests.*",
    "Merge*",
    "MergeByKey*",
    "Mr*Tests.*",
    "Partition*",
    "PartitionPoint*",
    "PermutationIteratorTests.*",
    "RandomTests.*",
    "Reduce*",
    "ReduceByKey*",
    "Remove*",
    "RemoveIf*",
    "Replace*",
    "ReverseIterator*",
    "Scan*",
    "ScanByKey*",
    "Scatter*",
    "Sequence*",
    "SetDifference*",
    "SetIntersection*",
    "SetSymmetricDifference*",
    "Shuffle*",
    "Sort*",
    "StableSort*",
    "StableSortByKey*",
    "Tabulate*",
    "TestBijectionLength",
    "TestHipThrustCopy.DeviceToDevice",
    "Transform*",
    "TransformIteratorTests.*",
    "TransformReduce*",
    "TransformScan*",
    "UninitializedCopy*",
    "UninitializedFill*",
    "Unique*",
    "Vector*",
    "VectorAllocatorTests.*",
    "ZipIterator*",
]

# Some platforms are less capable than others.
ctest_parallel_count = 8
if AMDGPU_FAMILIES == "gfx1153":
    ctest_parallel_count = 4

cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/rocthrust",
    "--output-on-failure",
    "--parallel",
    f"{ctest_parallel_count}",
    "--timeout",
    "300",
    "--repeat",
    "until-pass:6",
]

# If smoke tests are enabled, we run smoke tests only.
# Otherwise, we run the normal test suite
environ_vars = os.environ.copy()
test_type = os.getenv("TEST_TYPE", "full")
if test_type == "smoke":
    environ_vars["GTEST_FILTER"] = ":".join(SMOKE_TESTS)

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)
