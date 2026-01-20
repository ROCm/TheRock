import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

# Resolve absolute paths
THEROCK_BIN_PATH = Path(THEROCK_BIN_DIR).resolve()
THEROCK_PATH = THEROCK_BIN_PATH.parent
THEROCK_LIB_PATH = str(THEROCK_PATH / "lib")

print(f"THEROCK_PATH: {THEROCK_PATH}")
print(f"THEROCK_BIN_PATH: {THEROCK_BIN_PATH}")
print(f"THEROCK_LIB_PATH: {THEROCK_PATH}")

# Set up ROCM_PATH
environ_vars = os.environ.copy()
environ_vars["ROCM_PATH"] = str(THEROCK_PATH)

# Set up PYTHONPATH (for test_utils.py)
old_pythonpath = os.getenv("PYTHONPATH", "")
module_dir = f"{THEROCK_PATH}/libexec/rocprofiler-compute/tests"
print(f"Module Directory: {module_dir}")
if old_pythonpath:
    environ_vars["PYTHONPATH"] = f"{module_dir}:{old_pythonpath}"
else:
    environ_vars["PYTHONPATH"] = module_dir

# Set up PATH
old_path = os.getenv("PATH", "")
rocm_bin = str(THEROCK_BIN_PATH)
print(f"ROCm Bin Directory: {rocm_bin}")
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

# Print out all env vars
for key, value in environ_vars.items():
    print(f"{key}: {value}")

# Run tests
cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_PATH}/libexec/rocprofiler-compute",
    "--output-on-failure",
    "--timeout",
    "1800",
]
logging.info(f"++ Exec [{THEROCK_PATH}]$ {shlex.join(cmd)}")
subprocess.run(
    cmd,
    cwd=THEROCK_PATH,
    check=True,
    env=environ_vars,
)
