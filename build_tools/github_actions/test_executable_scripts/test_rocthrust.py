import logging
import os
import platform
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
if AMDGPU_FAMILIES == "gfx1152":
    ctest_parallel_count = 4
elif AMDGPU_FAMILIES == "gfx1153":
    ctest_parallel_count = 4

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
exe_dir = rocm_base / "bin" / "rocthrust"

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
    f"{THEROCK_BIN_DIR}/rocthrust",
    "--output-on-failure",
    "--parallel",
    f"{ctest_parallel_count}",
    "--resource-spec-file",
    resource_spec_file,
    "--timeout",
    "300",
]

# If smoke tests are enabled, we run smoke tests only.
# Otherwise, we run the normal test suite
environ_vars = os.environ.copy()
test_type = os.getenv("TEST_TYPE", "full")
if test_type == "smoke":
    environ_vars["GTEST_FILTER"] = ":".join(SMOKE_TESTS)

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=environ_vars)
