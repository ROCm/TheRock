import logging
import os
import shlex
import subprocess
from pathlib import Path

# Resolve paths
THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

THEROCK_BIN_PATH = Path(THEROCK_BIN_DIR).resolve()
THEROCK_PATH = THEROCK_BIN_PATH.parent
THEROCK_LIB_PATH = str(THEROCK_PATH / "lib")
ROCPROFILER_COMPUTE_DIRECTORY = f"{THEROCK_PATH}" / "libexec" / "rocprofiler-compute"

# Set up ROCM_PATH
environ_vars = os.environ.copy()
environ_vars["ROCM_PATH"] = str(THEROCK_PATH)

# Set up PYTHONPATH
old_pythonpath = os.getenv("PYTHONPATH", "")
module_dir = f"{ROCPROFILER_COMPUTE_DIRECTORY}" / "tests"
if old_pythonpath:
    environ_vars["PYTHONPATH"] = f"{module_dir}:{old_pythonpath}"
else:
    environ_vars["PYTHONPATH"] = module_dir

# Set up PATH
old_path = os.getenv("PATH", "")
rocm_bin = str(THEROCK_BIN_PATH)
if old_path:
    environ_vars["PATH"] = f"{rocm_bin}:{old_path}"
else:
    environ_vars["PATH"] = rocm_bin

# Set up LD_LIBRARY_PATH
old_ld_lib_path = os.getenv("LD_LIBRARY_PATH", "")
sysdeps_path = f"{THEROCK_LIB_PATH}/rocm_sysdeps/lib"
if old_ld_lib_path:
    environ_vars["LD_LIBRARY_PATH"] = (
        f"{THEROCK_LIB_PATH}:{sysdeps_path}:{old_ld_lib_path}"
    )
else:
    environ_vars["LD_LIBRARY_PATH"] = f"{THEROCK_LIB_PATH}:{sysdeps_path}"

# Set up excluded tests (include Jiras)
# AIPROFSDK-36: rocr issue causing test to fail
EXCLUDED_TESTS = [
    "test_profile_pc_sampling",
]

# Sharding
shard_index = int(os.getenv("SHARD_INDEX", "1")) - 1
total_shards = int(os.getenv("TOTAL_SHARDS", "1"))

# Smoke Tests Setup
SMOKE_TESTS = [
    "test_autogen_config",
    "test_utils",
    "test_num_xcds_cli_output",
    "test_num_xcds_spec_class",
    "test_L1_cache_counters",
    "test_analyze_workloads",
    "test_analyze_commands",
    "test_metric_validation",
    "test_profile_iteration_multiplexing_1",
]

# Run tests
cmd = [
    "ctest",
    "--test-dir",
    f"{ROCPROFILER_COMPUTE_DIRECTORY}",
    "--output-on-failure",
    "--verbose",
    "--exclude-regex",
    f"{"|".join(EXCLUDED_TESTS)}",
    "--tests-information",
    f"{shard_index},,{total_shards}",
]

# If smoke tests are enabled, we run smoke tests only.
# Otherwise, we run the normal test suite
test_type = os.getenv("TEST_TYPE", "full")
if test_type == "smoke":
    cmd.append("--tests-regex", "|".join(SMOKE_TESTS))

logging.info(f"++ Exec [{THEROCK_PATH}]$ {shlex.join(cmd)}")
subprocess.run(
    cmd,
    cwd=THEROCK_PATH,
    check=True,
    env=environ_vars,
)
